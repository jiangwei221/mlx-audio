# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.qwen3_tts.qwen3_tts import (
    IncrementalCustomVoiceSession,
    Model,
    TokenizationMovedCommittedBoundaryError,
    mel_spectrogram,
)
from mlx_audio.tts.models.qwen3_tts.speaker_encoder import (
    TimeDelayNetBlock,
    reflect_pad_1d,
)


class TestReflectPad1d(unittest.TestCase):
    """Tests for reflect_pad_1d helper function."""

    def test_no_padding(self):
        """Test that pad=0 returns the input unchanged."""
        x = mx.ones((1, 5, 3))
        result = reflect_pad_1d(x, pad=0)
        np.testing.assert_array_equal(np.array(result), np.array(x))

    def test_pad_1(self):
        """Test reflect padding with pad=1."""
        # Input: [1, 5, 1] with values [0, 1, 2, 3, 4]
        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        result = reflect_pad_1d(x, pad=1)

        # Reflect pad=1: left mirrors x[1], right mirrors x[-2]
        # Expected: [1, 0, 1, 2, 3, 4, 3]
        expected = np.array([[[1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [3.0]]])
        np.testing.assert_array_equal(np.array(result), expected)

    def test_pad_2(self):
        """Test reflect padding with pad=2."""
        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        result = reflect_pad_1d(x, pad=2)

        # Reflect pad=2: left mirrors x[1:3] reversed, right mirrors x[-3:-1] reversed
        # Left: x[1:3] = [1,2] reversed = [2,1]
        # Right: x[-3:-1] = [2,3] reversed = [3,2]
        # Expected: [2, 1, 0, 1, 2, 3, 4, 3, 2]
        expected = np.array(
            [[[2.0], [1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [3.0], [2.0]]]
        )
        np.testing.assert_array_equal(np.array(result), expected)

    def test_output_shape(self):
        """Test that output shape is [batch, time + 2*pad, channels]."""
        batch, time, channels = 2, 10, 4
        pad = 3
        x = mx.random.normal((batch, time, channels))
        result = reflect_pad_1d(x, pad)
        self.assertEqual(result.shape, (batch, time + 2 * pad, channels))

    def test_multichannel(self):
        """Test that reflect padding works correctly across multiple channels."""
        # Each channel should be padded independently with the same pattern
        x = mx.array(
            [
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                ]
            ]
        )
        result = reflect_pad_1d(x, pad=1)
        result_np = np.array(result)

        # Channel 0: [2, 1, 2, 3, 4, 5, 4]
        expected_ch0 = [2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
        # Channel 1: [20, 10, 20, 30, 40, 50, 40]
        expected_ch1 = [20.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0]

        np.testing.assert_array_equal(result_np[0, :, 0], expected_ch0)
        np.testing.assert_array_equal(result_np[0, :, 1], expected_ch1)


class TestTimeDelayNetBlockReflectPadding(unittest.TestCase):
    """Tests for TimeDelayNetBlock reflect padding behavior."""

    def test_output_shape_preserves_time(self):
        """Test that TimeDelayNetBlock with reflect padding preserves time dimension."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 3, 1
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 20
        x = mx.random.normal((batch, in_channels, time))  # NCL format
        out = block(x)

        # With reflect padding, output time should equal input time
        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_output_shape_with_dilation(self):
        """Test that dilated convolution with reflect padding preserves time."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 3, 2
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 20
        x = mx.random.normal((batch, in_channels, time))
        out = block(x)

        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_output_shape_kernel5_dilation2(self):
        """Test larger kernel with dilation preserves time."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 5, 2
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 30
        x = mx.random.normal((batch, in_channels, time))
        out = block(x)

        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_kernel1_no_padding(self):
        """Test that kernel_size=1 results in no padding."""
        block = TimeDelayNetBlock(16, 32, kernel_size=1, dilation=1)
        self.assertEqual(block.pad, 0)

    def test_pad_calculation(self):
        """Test that padding is computed correctly for various kernel/dilation combos."""
        # kernel=3, dilation=1 -> pad = (3-1)*1//2 = 1
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=1)
        self.assertEqual(block.pad, 1)

        # kernel=3, dilation=2 -> pad = (3-1)*2//2 = 2
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=2)
        self.assertEqual(block.pad, 2)

        # kernel=5, dilation=1 -> pad = (5-1)*1//2 = 2
        block = TimeDelayNetBlock(16, 32, kernel_size=5, dilation=1)
        self.assertEqual(block.pad, 2)

        # kernel=5, dilation=3 -> pad = (5-1)*3//2 = 6
        block = TimeDelayNetBlock(16, 32, kernel_size=5, dilation=3)
        self.assertEqual(block.pad, 6)

    def test_output_is_relu_activated(self):
        """Test that output values are non-negative (ReLU applied)."""
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=1)

        x = mx.random.normal((1, 16, 50))
        out = block(x)
        out_np = np.array(out)

        self.assertTrue(np.all(out_np >= 0), "Output should be non-negative after ReLU")


class TestMelSpectrogram(unittest.TestCase):
    """Tests for mel_spectrogram verifying correct parameters.

    Uses snapshot values from the known-correct implementation to detect
    if mel_scale, norm, or center/reflect padding is changed.
    """

    def _get_random_audio(self):
        """Get deterministic random audio (seed=42, 12000 samples)."""
        np.random.seed(42)
        return np.random.randn(12000).astype(np.float32)

    def test_output_shape_with_padding(self):
        """Test that manual padding + center=False produces 46 frames for 12000 samples."""
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)

        # manual 384 pad + center=False
        # padded_len = 12000 + 2*384 = 12768
        # frames = 1 + (12768 - 1024) // 256 = 46
        self.assertEqual(
            mel.shape,
            (1, 46, 128),
            "mel_spectrogram must use manual padding without center padding"
            f"Got shape {tuple(mel.shape)}, expected (1, 46, 128).",
        )

    def test_slaney_norm_values(self):
        """Test that slaney norm is applied (not norm=None).

        Without slaney norm, values would be ~3-5 (positive).
        With slaney norm, values are around -1 to 0.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Without norm="slaney", mel values are much higher (~3-5 range)
        self.assertLess(
            mel_np.mean(),
            1.0,
            "mel_spectrogram must use norm='slaney' in mel_filters(). "
            f"Got mean={mel_np.mean():.2f}, expected ~-0.37. "
            "Values > 1.0 indicate norm=None is being used.",
        )

        # Reference values from official Qwen3-TTS (slaney norm + slaney scale)
        expected_frame0 = np.array(
            [-0.21803714, 0.06630915, -0.31858957, -0.02480409, -0.4512914, -0.5911693]
        )
        actual_frame0 = mel_np[0, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_frame0,
            expected_frame0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use norm='slaney' in mel_filters(). "
            "These values are specific to slaney-normalized filterbank.",
        )

    def test_slaney_mel_scale(self):
        """Test that slaney mel scale is used (not htk).

        HTK scale distributes mel bins differently, producing different values.
        With slaney scale, frame 0 bin 0 ≈ -0.22. With htk, it's ≈ -0.53.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # With htk scale, low-frequency bins shift significantly
        # slaney: frame[0][0] ≈ -0.22, htk: frame[0][0] ≈ -0.53
        self.assertAlmostEqual(
            float(mel_np[0, 0]),
            -0.21803714,
            places=2,
            msg="mel_spectrogram must use mel_scale='slaney' in mel_filters(). "
            f"Got frame[0][bin 0]={mel_np[0, 0]:.4f}, expected ≈-0.22. "
            "A value of ≈-0.53 indicates mel_scale='htk' is being used.",
        )

        # Frame 23, selected bins - these values are specific to slaney scale
        expected_frame23 = np.array(
            [0.08127937, 0.4368576, 0.43200976, -0.7714137, -0.24601418, 0.04274124]
        )
        actual_frame23 = mel_np[23, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_frame23,
            expected_frame23,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use mel_scale='slaney' in mel_filters(). "
            "HTK scale distributes mel bins differently and produces wrong values.",
        )

    def test_reflect_padding_values(self):
        """Test that reflect padding produces correct boundary frame values.

        Without reflect padding, the first and last frames would have different
        values because the signal edges are handled differently.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Last frame values - sensitive to padding mode
        expected_last = np.array(
            [-0.16861804, 0.0474052, -0.3970174, -0.01738772, -0.28846806, -0.10941511]
        )
        actual_last = mel_np[-1, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_last,
            expected_last,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use reflect padding. "
            "Boundary frames are sensitive to the padding mode used before STFT.",
        )

    def test_sine_wave_mel_bins(self):
        """Test that a 1kHz sine wave activates the correct mel bins.

        This verifies both the mel scale and filterbank norm are correct,
        since wrong parameters would shift energy to different bins.
        """
        t = np.arange(12000, dtype=np.float32) / 24000.0
        audio = mx.array(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Frame 0 values for a 1kHz sine - specific to slaney scale + slaney norm
        expected = np.array(
            [
                -1.2959518,
                -1.2937515,
                -1.2902284,
                -1.2074544,
                -0.9268621,
                -2.3822036,
                -5.331841,
                -5.33782,
            ]
        )
        actual = mel_np[0, [0, 1, 2, 10, 20, 63, 126, 127]]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use mel_scale='slaney' and norm='slaney'. "
            "A 1kHz sine wave should produce these specific bin activations "
            "with slaney-scale mel filterbank.",
        )

    def test_overall_statistics(self):
        """Test overall mean and std match expected values."""
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)

        np.testing.assert_allclose(
            mel_np.mean(),
            -0.37329558,
            rtol=1e-3,
            err_msg="mel_spectrogram output mean should be ~-0.37 with correct params. "
            f"Got mean={mel_np.mean():.4f}. A positive mean (~2.5) indicates norm=None.",
        )
        np.testing.assert_allclose(
            mel_np.std(),
            0.37445435,
            rtol=1e-3,
            err_msg="mel_spectrogram output std should be ~0.37 with correct params. "
            f"Got std={mel_np.std():.4f}.",
        )


class _FakeEmbedding:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def __call__(self, token_ids: mx.array) -> mx.array:
        ids = token_ids
        if ids.ndim == 1:
            ids = ids[None, :]
        values = ids.astype(mx.float32)[..., None]
        return mx.broadcast_to(values, (ids.shape[0], ids.shape[1], self.hidden_size))


class _FakeCodePredictor:
    def __init__(self, vocab_size: int, hidden_size: int, num_code_groups: int):
        self.vocab_size = vocab_size
        self.codec_embedding = [
            _FakeEmbedding(hidden_size) for _ in range(num_code_groups - 1)
        ]

    def make_cache(self):
        return {}

    def __call__(self, input_embeds: mx.array, cache=None, generation_step: int = 0):
        batch, seq_len, _ = input_embeds.shape
        logits = mx.zeros((batch, seq_len, self.vocab_size), dtype=mx.float32)
        return logits, cache, generation_step + 1


class _FakeTalker:
    def __init__(self, vocab_size: int, hidden_size: int, num_code_groups: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._input_embedding = _FakeEmbedding(hidden_size)
        self.code_predictor = _FakeCodePredictor(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_code_groups=num_code_groups,
        )

    def make_cache(self):
        return {}

    def get_input_embeddings(self):
        return self._input_embedding

    def __call__(self, input_embeds: mx.array, cache=None):
        batch, seq_len, _ = input_embeds.shape
        logits = mx.zeros((batch, seq_len, self.vocab_size), dtype=mx.float32)
        hidden = mx.zeros((batch, seq_len, self.hidden_size), dtype=mx.float32)
        return logits, hidden


class _FakeIncrementalModel:
    def __init__(
        self,
        tokenization_map: dict,
        primary_tokens: list[int],
        eos_token_id: int = 63,
        codec_pad_id: int = 60,
        codec_bos_id: int = 61,
        num_code_groups: int = 2,
        hidden_size: int = 8,
        vocab_size: int = 64,
    ):
        self._tokenization_map = tokenization_map
        self._primary_tokens = list(primary_tokens)
        self._secondary_token = 7
        self._hidden_size = hidden_size

        self.config = SimpleNamespace(
            talker_config=SimpleNamespace(
                codec_eos_token_id=eos_token_id,
                codec_pad_id=codec_pad_id,
                codec_bos_id=codec_bos_id,
                num_code_groups=num_code_groups,
                vocab_size=vocab_size,
            )
        )
        self.sample_rate = 24000
        self.speech_tokenizer = SimpleNamespace(decode_upsample_rate=1)
        self.talker = _FakeTalker(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_code_groups=num_code_groups,
        )

    def _build_custom_voice_session_context(
        self,
        speaker: str,
        language: str = "auto",
        instruct: str | None = None,
    ):
        base = mx.zeros((1, 1, self._hidden_size), dtype=mx.float32)
        return {
            "prefill_prefix": base,
            "codec_suffix": base,
            "tts_pad_embed": base,
            "tts_eos_embed": mx.ones((1, 1, self._hidden_size), dtype=mx.float32),
            "suppress_tokens": mx.array([], dtype=mx.int32),
        }

    def _tokenize_chat_body_ids(self, text: str):
        if text not in self._tokenization_map:
            return []
        return list(self._tokenization_map[text])

    def _embed_body_text_token_ids(self, token_ids: list[int]):
        return [
            mx.full((1, 1, self._hidden_size), float(token_id), dtype=mx.float32)
            for token_id in token_ids
        ]

    def _sample_token(self, logits: mx.array, suppress_tokens=None, eos_token_id=None, **_):
        if suppress_tokens is not None:
            suppressed = set(int(t) for t in suppress_tokens)
            token = None
            while self._primary_tokens:
                candidate = int(self._primary_tokens.pop(0))
                if candidate in suppressed:
                    continue
                token = candidate
                break
            if token is None:
                if eos_token_id is not None and int(eos_token_id) not in suppressed:
                    token = int(eos_token_id)
                else:
                    token = int(self._secondary_token)
            return mx.array([[token]], dtype=mx.int32)
        return mx.array([[self._secondary_token]], dtype=mx.int32)

    def _decode_chunk(self, codes: mx.array, chunk_tokens: int = 100):
        del chunk_tokens
        first_codebook = np.array(codes)[0, :, 0].astype(np.float32)
        return mx.array(first_codebook)


class TestQwen3TTSIncrementalCustomVoiceSession(unittest.TestCase):
    def _new_session(
        self,
        model: _FakeIncrementalModel,
        stable_tail_tokens: int = 1,
        waiting_text_strategy: str = "pause",
        catchup_target_codec_per_text: float = 2.0,
        catchup_max_steps_per_wait: int = 2,
    ):
        return IncrementalCustomVoiceSession(
            model=model,
            speaker="vivian",
            stable_tail_tokens=stable_tail_tokens,
            streaming_interval=0.08,  # Force 1 token streaming chunks in tests.
            waiting_text_strategy=waiting_text_strategy,
            catchup_target_codec_per_text=catchup_target_codec_per_text,
            catchup_max_steps_per_wait=catchup_max_steps_per_wait,
            max_codec_steps_total=32,
        )

    def test_basic_incremental_flow_wait_resume_and_finalize(self):
        tokenization_map = {
            "A": [11, 12],
            "AB": [11, 12, 13, 14],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[21, 22, 23, 63],
        )
        session = self._new_session(model, stable_tail_tokens=1)

        session.append_text("A")
        first_results = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(first_results), 1)
        self.assertTrue(session.is_waiting_text())
        self.assertTrue(session.status()["paused_for_text"])
        self.assertEqual(session.status()["consumed_text_tokens"], 1)

        first_hashes = [hash(np.array(r.audio).tobytes()) for r in first_results]

        session.append_text("B")
        second_results = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(second_results), 1)
        self.assertFalse(session.is_waiting_text())
        self.assertFalse(session.status()["paused_for_text"])
        self.assertEqual(
            first_hashes, [hash(np.array(r.audio).tobytes()) for r in first_results]
        )

        session.finalize_text()
        for _ in range(10):
            if session.is_finished():
                break
            list(session.pump(max_codec_steps=1))
        self.assertTrue(session.is_finished())

    def test_tokenization_change_in_unconsumed_region_is_allowed(self):
        tokenization_map = {
            "M": [10, 20],
            "MN": [10, 99, 100],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[31, 32, 63],
        )
        session = self._new_session(model, stable_tail_tokens=1)

        session.append_text("M")
        list(session.pump(max_codec_steps=1))
        session.append_text("N")

        self.assertEqual(session.consumed_text_tokens, 1)
        self.assertEqual(session.all_text_ids[:2], [10, 99])
        self.assertGreater(session.status()["pending_tokens"], 0)

    def test_tokenization_change_crossing_committed_boundary_raises(self):
        tokenization_map = {
            "X": [10, 20],
            "XY": [77, 20, 30],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[41, 63],
        )
        session = self._new_session(model, stable_tail_tokens=1)

        session.append_text("X")
        list(session.pump(max_codec_steps=1))

        with self.assertRaises(TokenizationMovedCommittedBoundaryError):
            session.append_text("Y")

    def test_waiting_text_then_resume_after_append(self):
        tokenization_map = {
            "a": [1, 2],
            "ab": [1, 2, 3, 4, 5],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[51, 52, 63],
        )
        session = self._new_session(model, stable_tail_tokens=4)

        session.append_text("a")
        self.assertEqual(list(session.pump(max_codec_steps=1)), [])
        self.assertTrue(session.is_waiting_text())
        self.assertFalse(session.is_finished())

        session.append_text("b")
        resumed = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(resumed), 1)
        self.assertTrue(session.status()["paused_for_text"])

    def test_pause_strategy_does_not_advance_codec_while_waiting(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[11, 12, 13, 14],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        self.assertTrue(session.is_waiting_text())
        self.assertTrue(session.status()["paused_for_text"])

        before = len(session.generated_codes)
        list(session.pump(max_codec_steps=3))
        after = len(session.generated_codes)
        self.assertEqual(before, after)
        self.assertTrue(session.is_waiting_text())
        self.assertTrue(session.status()["paused_for_text"])

    def test_pause_strategy_resumes_without_cache_reset(self):
        tokenization_map = {
            "A": [10, 11],
            "AB": [10, 11, 12, 13],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[21, 22, 23, 24, 25],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        cache_id = id(session.cache)
        self.assertTrue(session.is_waiting_text())
        self.assertTrue(session.status()["paused_for_text"])

        session.append_text("B")
        resumed = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(resumed), 1)
        self.assertEqual(cache_id, id(session.cache))
        self.assertFalse(session.status()["paused_for_text"])

    def test_pad_strategy_keeps_backward_compatible_waiting_drive(self):
        tokenization_map = {
            "A": [31, 32],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[41, 42, 43, 44],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pad",
        )

        list(session.pump(max_codec_steps=1))
        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        self.assertTrue(session.is_waiting_text())
        self.assertFalse(session.status()["paused_for_text"])
        before = len(session.generated_codes)
        second = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(second), 1)
        self.assertEqual(len(session.generated_codes), before + 1)

    def test_pause_catchup_advances_within_budget_then_hard_pauses(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[31, 32, 33, 34, 35],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause_catchup",
            catchup_target_codec_per_text=10.0,
            catchup_max_steps_per_wait=2,
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=16))
        self.assertTrue(session.is_waiting_text())
        self.assertTrue(session.status()["paused_for_text"])
        self.assertEqual(len(session.generated_codes), 3)  # 1 text step + 2 catchup steps
        self.assertEqual(session.status()["catchup_steps_total"], 2)
        self.assertEqual(session.status()["catchup_steps_current_wait"], 2)
        self.assertEqual(session.status()["wait_pause_step_target"], 3)

    def test_pause_catchup_lag_zero_pauses_immediately(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[41, 42, 43],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause_catchup",
            catchup_target_codec_per_text=0.5,
            catchup_max_steps_per_wait=2,
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=8))
        self.assertTrue(session.is_waiting_text())
        self.assertEqual(len(session.generated_codes), 1)
        self.assertEqual(session.status()["catchup_steps_total"], 0)
        self.assertEqual(session.status()["catchup_steps_current_wait"], 0)
        self.assertEqual(session.status()["wait_pause_step_target"], 1)

    def test_pause_catchup_resumes_without_cache_reset(self):
        tokenization_map = {
            "A": [10, 11],
            "AB": [10, 11, 12, 13],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[51, 52, 53, 54, 55],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause_catchup",
            catchup_target_codec_per_text=10.0,
            catchup_max_steps_per_wait=2,
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=8))
        cache_id = id(session.cache)
        self.assertTrue(session.is_waiting_text())

        session.append_text("B")
        resumed = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(resumed), 1)
        self.assertEqual(cache_id, id(session.cache))
        self.assertFalse(session.status()["paused_for_text"])
        self.assertIsNone(session.status()["wait_pause_step_target"])
        self.assertEqual(session.status()["catchup_steps_current_wait"], 0)

    def test_pause_catchup_waiting_pump_does_not_emit_duplicate_chunks(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[61, 62, 63, 64, 65],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause_catchup",
            catchup_target_codec_per_text=10.0,
            catchup_max_steps_per_wait=2,
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=16))
        self.assertTrue(session.is_waiting_text())

        decoded_before = session.decoded_tokens
        generated_before = len(session.generated_codes)
        second = list(session.pump(max_codec_steps=3))
        self.assertEqual(second, [])
        self.assertEqual(decoded_before, session.decoded_tokens)
        self.assertEqual(generated_before, len(session.generated_codes))

    def test_waiting_pump_does_not_emit_duplicate_chunks(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[61, 62, 63],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        first = list(session.pump(max_codec_steps=1))
        self.assertEqual(len(first), 1)
        self.assertTrue(session.is_waiting_text())

        decoded_before = session.decoded_tokens
        generated_before = len(session.generated_codes)

        second = list(session.pump(max_codec_steps=3))
        self.assertEqual(second, [])
        self.assertEqual(decoded_before, session.decoded_tokens)
        self.assertEqual(generated_before, len(session.generated_codes))

    def test_stable_tail_defers_last_token_to_next_append(self):
        tokenization_map = {
            "A": [11, 12],
            "AB": [11, 12, 13],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[71, 72, 73, 63],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        self.assertEqual(session.consumed_text_tokens, 1)
        self.assertEqual(session.all_text_ids, [11, 12])

        session.append_text("B")
        self.assertGreaterEqual(len(session._pending_token_ids), 1)
        self.assertEqual(session._pending_token_ids[0], 12)

    def test_stable_tail_zero_consumes_all_tokens_without_deferral(self):
        tokenization_map = {
            "A": [21, 22],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[91, 92, 63],
        )
        session = self._new_session(
            model,
            stable_tail_tokens=0,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=2))
        self.assertEqual(session.consumed_text_tokens, 2)
        self.assertEqual(session.status()["pending_tokens"], 0)
        self.assertTrue(session.is_waiting_text())

    def test_first_codebook_sampling_suppresses_codec_pad_and_bos(self):
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[60, 61, 45, 63],
            codec_pad_id=60,
            codec_bos_id=61,
            eos_token_id=63,
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        self.assertEqual(int(session.generated_codes[0][0, 0]), 45)

    def test_eos_is_blocked_before_finalize_and_allowed_after_finalize(self):
        eos = 63
        tokenization_map = {
            "A": [1, 2],
        }
        model = _FakeIncrementalModel(
            tokenization_map=tokenization_map,
            primary_tokens=[eos, 81, eos],
            eos_token_id=eos,
        )
        session = self._new_session(
            model,
            stable_tail_tokens=1,
            waiting_text_strategy="pause",
        )

        session.append_text("A")
        list(session.pump(max_codec_steps=1))
        self.assertFalse(session.is_finished())
        self.assertEqual(int(session.generated_codes[0][0, 0]), 81)

        session.finalize_text()
        for _ in range(8):
            if session.is_finished():
                break
            list(session.pump(max_codec_steps=1))
        self.assertTrue(session.is_finished())

    def test_generate_custom_voice_incremental_wrapper(self):
        class _MockSession:
            def __init__(self):
                self.appended = []
                self.finalized = False
                self.finished = False
                self._pump_calls = 0

            def append_text(self, chunk: str):
                self.appended.append(chunk)

            def finalize_text(self):
                self.finalized = True

            def pump(self, max_codec_steps=None):
                del max_codec_steps
                self._pump_calls += 1
                if self._pump_calls <= len(self.appended):
                    yield f"chunk-{self._pump_calls}"
                    return
                if self.finalized and not self.finished:
                    self.finished = True
                    yield "final"

            def is_waiting_text(self):
                return False

            def is_finished(self):
                return self.finished

        mock_session = _MockSession()
        fake_model = SimpleNamespace(
            start_custom_voice_session=MagicMock(return_value=mock_session)
        )

        outputs = list(
            Model.generate_custom_voice_incremental(
                fake_model,
                text_chunks=["first", "second"],
                speaker="vivian",
                waiting_text_strategy="pad",
                catchup_target_codec_per_text=2.5,
                catchup_max_steps_per_wait=4,
            )
        )

        self.assertEqual(outputs, ["chunk-1", "chunk-2", "final"])
        self.assertEqual(mock_session.appended, ["first", "second"])
        self.assertTrue(mock_session.finalized)
        fake_model.start_custom_voice_session.assert_called_once()
        self.assertEqual(
            fake_model.start_custom_voice_session.call_args.kwargs[
                "waiting_text_strategy"
            ],
            "pad",
        )
        self.assertEqual(
            fake_model.start_custom_voice_session.call_args.kwargs[
                "catchup_target_codec_per_text"
            ],
            2.5,
        )
        self.assertEqual(
            fake_model.start_custom_voice_session.call_args.kwargs[
                "catchup_max_steps_per_wait"
            ],
            4,
        )

    def test_generate_custom_voice_compatibility_smoke(self):
        fake_model = SimpleNamespace(
            config=SimpleNamespace(tts_model_type="custom_voice", tts_model_size="1b7"),
            supported_speakers=["Vivian"],
            _generate_with_instruct=MagicMock(return_value=iter(["ok"])),
        )

        result = list(
            Model.generate_custom_voice(
                fake_model,
                text="hello",
                speaker="vivian",
                language="auto",
                instruct=None,
            )
        )

        self.assertEqual(result, ["ok"])
        fake_model._generate_with_instruct.assert_called_once()


if __name__ == "__main__":
    unittest.main()
