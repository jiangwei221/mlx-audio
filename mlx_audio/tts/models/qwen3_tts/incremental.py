"""Local, single-owner incremental CustomVoice session.

This intentionally small implementation is for interactive applications that
feed one Qwen3-TTS request from an LLM text stream.  It is not a batching or
network-session API: the upstream decoder has shared streaming state, so only
one active session may use a loaded model at a time.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Generator, List, Optional

import mlx.core as mx

from mlx_audio.tts.models.base import GenerationResult

from .text_commit import (
    CommitPolicy,
    TextCommitter,
    TokenizationMovedCommittedBoundaryError,
)

if TYPE_CHECKING:
    from .qwen3_tts import Model


class IncrementalCustomVoiceSession:
    _TOKENS_PER_SECOND = 12.5

    def __init__(
        self, model: "Model", speaker: str, language: str, instruct: Optional[str],
        commit_policy: str, streaming_interval: float, temperature: float,
        top_k: int, top_p: float, repetition_penalty: float,
        max_codec_steps_total: int,
    ) -> None:
        if streaming_interval <= 0:
            raise ValueError("streaming_interval must be > 0")
        if max_codec_steps_total <= 0:
            raise ValueError("max_codec_steps_total must be > 0")
        self.model = model
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.repetition_penalty = repetition_penalty
        self.max_codec_steps_total = max_codec_steps_total
        self._chunk_size = max(1, int(streaming_interval * self._TOKENS_PER_SECOND))
        self._context = model._build_incremental_custom_voice_context(speaker, language, instruct)
        self._config = model.config.talker_config
        self._sample_rate = int(model.sample_rate)
        self._committer = TextCommitter(
            model._tokenize_incremental_body_with_offsets,
            policy=CommitPolicy(commit_policy),
        )
        self.full_text = ""
        self.all_text_ids: List[int] = []
        self.consumed_text_tokens = 0
        self._pending_ids: Deque[int] = deque()
        self._pending_embeds: Deque[mx.array] = deque()
        self._generated_codes: List[mx.array] = []
        self._generated_first_codes: List[int] = []
        self._decoded_tokens = 0
        self._cache = None
        self._next_input: Optional[mx.array] = None
        self._next_codec_embed: Optional[mx.array] = None
        self._finalized = False
        self._text_eos_consumed = False
        self._finished = False
        self._waiting = False
        self._closed = False
        self._codec_steps = 0
        self._decoder_claimed = False

    def append_text(self, chunk: str) -> None:
        if self._finalized or self._closed:
            raise RuntimeError("session input is closed")
        if not isinstance(chunk, str):
            raise TypeError("text chunk must be a str")
        if not chunk:
            return
        proposal = self._committer.prepare_append(chunk)
        new_ids = proposal.sealed_token_ids[len(self.all_text_ids) :]
        new_embeds = self.model._embed_incremental_body(list(new_ids))
        # The embedding evaluation above can fail.  Only now do we mutate the
        # text snapshot and queues, preserving append atomicity.
        self._committer.commit(proposal)
        self.full_text = self._committer.snapshot.raw_text
        self.all_text_ids = list(proposal.sealed_token_ids)
        self._pending_ids.extend(new_ids)
        self._pending_embeds.extend(new_embeds)

    def finalize_text(self) -> None:
        if self._finalized:
            return
        proposal = self._committer.prepare_finalize()
        new_ids = proposal.sealed_token_ids[len(self.all_text_ids) :]
        new_embeds = self.model._embed_incremental_body(list(new_ids))
        self._committer.commit(proposal)
        self.full_text = self._committer.snapshot.raw_text
        self.all_text_ids = list(proposal.sealed_token_ids)
        self._pending_ids.extend(new_ids)
        self._pending_embeds.extend(new_embeds)
        self._finalized = True

    def is_finished(self) -> bool:
        return self._finished

    def is_waiting_text(self) -> bool:
        return self._waiting and not self._finished

    def status(self) -> dict:
        return {"finished": self._finished, "finalized": self._finalized, "waiting_text": self._waiting, "consumed_text_tokens": self.consumed_text_tokens, "pending_tokens": len(self._pending_ids), "codec_tokens": len(self._generated_codes)}

    def pump(self, max_codec_steps: Optional[int] = None) -> Generator[GenerationResult, None, None]:
        if max_codec_steps is not None and max_codec_steps <= 0:
            raise ValueError("max_codec_steps must be > 0")
        if self._finished or self._closed:
            return
        self._claim_decoder()
        try:
            self._resume_or_bootstrap()
            steps = 0
            while self._next_input is not None and not self._finished:
                if max_codec_steps is not None and steps >= max_codec_steps:
                    break
                codes, codec_embed = self._run_step()
                if codes is None:
                    self._finished = True
                    yield from self._emit_remaining(final=True)
                    break
                self._generated_codes.append(codes)
                self._generated_first_codes.append(int(codes[0, 0]))
                steps += 1
                if len(self._generated_codes) - self._decoded_tokens >= self._chunk_size:
                    yield self._decode_new(final=False)
                self._prepare_next(codec_embed)
            if self._waiting:
                yield from self._emit_remaining(final=False)
        finally:
            # The decoder state remains claimed between calls; it is released
            # only by close() after reset, preventing another session from
            # corrupting this session's audio stream.
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._decoder_claimed:
            self.model.speech_tokenizer.decoder.reset_streaming_state()
            setattr(self.model, "_incremental_decoder_owner", None)
        self._decoder_claimed = False
        self._cache = None
        self._next_input = None

    def _claim_decoder(self) -> None:
        owner = getattr(self.model, "_incremental_decoder_owner", None)
        if owner is not None and owner is not self:
            raise RuntimeError("another incremental session is using this model's decoder")
        if not self._decoder_claimed:
            setattr(self.model, "_incremental_decoder_owner", self)
            self.model.speech_tokenizer.decoder.reset_streaming_state()
            self._decoder_claimed = True

    def _consume_text(self) -> Optional[mx.array]:
        if self._pending_embeds:
            self._pending_ids.popleft()
            self._committer.consume_ready(1)
            self.consumed_text_tokens += 1
            return self._pending_embeds.popleft()
        if self._finalized and not self._text_eos_consumed:
            self._text_eos_consumed = True
            return self._context["tts_eos_embed"]
        return None

    def _resume_or_bootstrap(self) -> None:
        if self._next_input is not None:
            return
        text_embed = self._consume_text()
        if text_embed is None:
            if self._finalized and not self.full_text:
                self._finished = True
            else:
                self._waiting = True
            return
        if self._cache is None:
            self._cache = self.model.talker.make_cache()
            self._next_input = mx.concatenate([self._context["prefill_prefix"], text_embed + self._context["codec_suffix"]], axis=1)
        else:
            self._next_input = text_embed + self._next_codec_embed
        mx.eval(self._next_input)
        self._waiting = False

    def _run_step(self):
        if self._codec_steps >= self.max_codec_steps_total:
            return None, None
        logits, hidden = self.model.talker(self._next_input, cache=self._cache)
        allow_eos = self._finalized and self._text_eos_consumed
        eos = self._config.codec_eos_token_id
        suppress = [i for i in range(self._config.vocab_size - 1024, self._config.vocab_size) if i != eos]
        if not allow_eos:
            suppress.append(eos)
        first = self.model._sample_token(logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, repetition_penalty=self.repetition_penalty, generated_tokens=self._generated_first_codes or None, suppress_tokens=suppress, eos_token_id=eos if allow_eos else None)
        if allow_eos and int(first[0, 0]) == eos:
            return None, None
        cache = self.model.talker.code_predictor.make_cache()
        tokens, code_hidden = [first], hidden[:, -1:, :]
        for index in range(self._config.num_code_groups - 1):
            input_embed = mx.concatenate([code_hidden, self.model.talker.get_input_embeddings()(first)], axis=1) if index == 0 else self.model.talker.code_predictor.codec_embedding[index - 1](tokens[-1])
            logits, cache, _ = self.model.talker.code_predictor(input_embed, cache=cache, generation_step=index)
            tokens.append(self.model._sample_token(logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p))
        codes = mx.concatenate(tokens, axis=1)
        codec_embed = self.model.talker.get_input_embeddings()(first)
        for index, token in enumerate(tokens[1:]):
            codec_embed = codec_embed + self.model.talker.code_predictor.codec_embedding[index](token)
        self._codec_steps += 1
        return codes, codec_embed

    def _prepare_next(self, codec_embed: mx.array) -> None:
        self._next_codec_embed = codec_embed
        text_embed = self._consume_text()
        if text_embed is None:
            if not self._finalized:
                self._next_input, self._waiting = None, True
                return
            text_embed = self._context["tts_pad_embed"]
        self._next_input = text_embed + codec_embed
        mx.eval(self._next_input)

    def _decode_new(self, final: bool) -> GenerationResult:
        codes = mx.stack(self._generated_codes[self._decoded_tokens :], axis=1)
        codes = mx.transpose(codes, (0, 2, 1))
        wav = self.model.speech_tokenizer.decoder.streaming_step(codes).squeeze(1)[0]
        mx.eval(wav)
        count = len(self._generated_codes) - self._decoded_tokens
        self._decoded_tokens = len(self._generated_codes)
        samples = int(wav.shape[0])
        return GenerationResult(wav, samples, self._sample_rate, 0, count, f"{samples / self._sample_rate:.2f}s", 0.0, {"tokens": count}, {"samples": samples}, 0.0, mx.get_peak_memory() / 1e9, True, final)

    def _emit_remaining(self, final: bool) -> Generator[GenerationResult, None, None]:
        if len(self._generated_codes) > self._decoded_tokens:
            yield self._decode_new(final)
