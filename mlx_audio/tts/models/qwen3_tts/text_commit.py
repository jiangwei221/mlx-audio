"""Transactional text commitment for incremental Qwen3-TTS sessions.

The talker cache is append-only, while BPE tokenization of the unfinished
right edge of a text stream is not.  This module keeps that mutable edge out
of the cache by exposing only token ids whose character spans are known to be
stable.  It deliberately has no MLX dependency so its invariants can be
tested without model weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class TextCommitError(RuntimeError):
    """Base error for incremental text commitment."""


class TokenizerOffsetError(TextCommitError):
    """The tokenizer did not provide a usable offset mapping."""


class TokenizationMovedCommittedBoundaryError(TextCommitError):
    """Retokenization attempted to rewrite text already made immutable."""


class TextInputLimitError(TextCommitError):
    """The stream exceeded a configured text buffering limit."""


class CommitPolicy(str, Enum):
    SAFE_WORD = "safe_word"
    SAFE_SENTENCE = "safe_sentence"
    LEGACY_TOKEN_TAIL = "legacy_token_tail"


class TokenEncoding(NamedTuple):
    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TextCommitSnapshot:
    raw_text: str
    sealed_char_end: int
    sealed_token_ids: tuple[int, ...]
    consumed_token_count: int
    ready_token_ids: tuple[int, ...]
    unstable_suffix: str
    version: int


@dataclass(frozen=True)
class AppendResult:
    accepted_chars: int
    newly_sealed_tokens: int
    sealed_tokens_total: int
    consumed_tokens_total: int
    buffered_chars: int


@dataclass(frozen=True)
class TextCommitProposal:
    """A fully validated candidate that has not mutated a committer yet."""

    base_version: int
    raw_text: str
    sealed_char_end: int
    sealed_token_ids: tuple[int, ...]
    accepted_chars: int


def _safe_word_end(text: str) -> int:
    """Return the start of the final whitespace-delimited lexical span.

    Keeping the separator itself is intentional: many BPE tokenizers fold a
    leading space into the following word token.  This is conservative for
    URLs, numbers and identifiers as they contain no whitespace.
    """

    if not text:
        return 0
    end = len(text)
    while end and text[end - 1].isspace():
        end -= 1
    if end == 0:
        return 0
    start = end
    while start and not text[start - 1].isspace():
        start -= 1
    # Do not seal the preceding separator either.  BPE vocabularies commonly
    # merge it with the next lexical span.
    while start and text[start - 1].isspace():
        start -= 1
    return start


def _safe_sentence_end(text: str) -> int:
    """Return the end of the last strong sentence boundary."""

    boundary = 0
    for index, char in enumerate(text[:-1]):
        if char in ".?!。！？\n":
            boundary = index + 1
    return boundary


class TextCommitter:
    """Own the mutable text tail and atomically seal stable token ids.

    ``tokenize`` receives the complete user text and must return ids with
    offsets in that same text.  Chat-template wrapping belongs in the adapter
    passed by the Qwen model; this prevents magic template slices here.
    """

    def __init__(
        self,
        tokenize: Callable[[str], TokenEncoding],
        *,
        policy: CommitPolicy | str = CommitPolicy.SAFE_WORD,
        legacy_tail_tokens: int = 4,
        max_uncommitted_chars: int = 4096,
        max_input_chars: int = 65536,
    ) -> None:
        self._tokenize = tokenize
        self.policy = CommitPolicy(policy)
        if legacy_tail_tokens < 0:
            raise ValueError("legacy_tail_tokens must be >= 0")
        if max_uncommitted_chars <= 0 or max_input_chars <= 0:
            raise ValueError("text limits must be > 0")
        self.legacy_tail_tokens = legacy_tail_tokens
        self.max_uncommitted_chars = max_uncommitted_chars
        self.max_input_chars = max_input_chars
        self._raw_text = ""
        self._sealed_char_end = 0
        self._sealed_ids: tuple[int, ...] = ()
        self._consumed = 0
        self._version = 0

    @property
    def snapshot(self) -> TextCommitSnapshot:
        return TextCommitSnapshot(
            raw_text=self._raw_text,
            sealed_char_end=self._sealed_char_end,
            sealed_token_ids=self._sealed_ids,
            consumed_token_count=self._consumed,
            ready_token_ids=self._sealed_ids[self._consumed :],
            unstable_suffix=self._raw_text[self._sealed_char_end :],
            version=self._version,
        )

    def append(self, delta: str) -> AppendResult:
        if not isinstance(delta, str):
            raise TypeError("text delta must be a str")
        if not delta:
            return self._result(0, 0)
        return self.commit(self.prepare_append(delta))

    def finalize(self) -> AppendResult:
        return self.commit(self.prepare_finalize())

    def prepare_append(self, delta: str) -> TextCommitProposal:
        if not isinstance(delta, str):
            raise TypeError("text delta must be a str")
        return self._prepare(self._raw_text + delta, accepted_chars=len(delta), finalized=False)

    def prepare_finalize(self) -> TextCommitProposal:
        return self._prepare(self._raw_text, accepted_chars=0, finalized=True)

    def commit(self, proposal: TextCommitProposal) -> AppendResult:
        if proposal.base_version != self._version:
            raise TextCommitError("text commit proposal is stale")
        previous_count = len(self._sealed_ids)
        self._raw_text = proposal.raw_text
        self._sealed_char_end = proposal.sealed_char_end
        self._sealed_ids = proposal.sealed_token_ids
        self._version += 1
        return self._result(proposal.accepted_chars, len(self._sealed_ids) - previous_count)

    def consume_ready(self, count: int = 1) -> tuple[int, ...]:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        end = min(self._consumed + count, len(self._sealed_ids))
        consumed = self._sealed_ids[self._consumed : end]
        self._consumed = end
        return consumed

    def _prepare(
        self, candidate_text: str, *, accepted_chars: int, finalized: bool
    ) -> TextCommitProposal:
        if len(candidate_text) > self.max_input_chars:
            raise TextInputLimitError("max_input_chars exceeded")
        encoding = self._validate_encoding(self._tokenize(candidate_text), candidate_text)
        candidate_end = len(candidate_text) if finalized else self._safe_end(candidate_text, encoding)
        sealed_count = self._sealed_token_count(encoding.offsets, candidate_end)
        candidate_ids = encoding.ids[:sealed_count]
        if candidate_ids[: len(self._sealed_ids)] != self._sealed_ids:
            raise TokenizationMovedCommittedBoundaryError(
                "retokenization changed an already sealed text prefix"
            )
        if self._consumed > len(candidate_ids):
            raise TokenizationMovedCommittedBoundaryError(
                "retokenization shortened text below the consumed boundary"
            )
        if len(candidate_text) - candidate_end > self.max_uncommitted_chars:
            raise TextInputLimitError("max_uncommitted_chars exceeded")

        return TextCommitProposal(
            base_version=self._version,
            raw_text=candidate_text,
            sealed_char_end=candidate_end,
            sealed_token_ids=candidate_ids,
            accepted_chars=accepted_chars,
        )

    def _safe_end(self, text: str, encoding: TokenEncoding) -> int:
        if self.policy is CommitPolicy.SAFE_WORD:
            return _safe_word_end(text)
        if self.policy is CommitPolicy.SAFE_SENTENCE:
            return _safe_sentence_end(text)
        stable_count = max(0, len(encoding.ids) - self.legacy_tail_tokens)
        if not stable_count:
            return 0
        return encoding.offsets[stable_count - 1][1]

    @staticmethod
    def _sealed_token_count(offsets: Sequence[tuple[int, int]], end: int) -> int:
        count = 0
        for start, token_end in offsets:
            if start < end and token_end <= end:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _validate_encoding(encoding: TokenEncoding, text: str) -> TokenEncoding:
        if len(encoding.ids) != len(encoding.offsets):
            raise TokenizerOffsetError("token ids and offsets have different lengths")
        previous_end = 0
        for start, end in encoding.offsets:
            if not 0 <= start <= end <= len(text) or start < previous_end:
                raise TokenizerOffsetError("tokenizer returned invalid or unordered offsets")
            previous_end = end
        return encoding

    def _result(self, accepted_chars: int, newly_sealed_tokens: int) -> AppendResult:
        return AppendResult(
            accepted_chars=accepted_chars,
            newly_sealed_tokens=newly_sealed_tokens,
            sealed_tokens_total=len(self._sealed_ids),
            consumed_tokens_total=self._consumed,
            buffered_chars=len(self._raw_text) - self._sealed_char_end,
        )
