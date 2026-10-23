"""Bounded codec positions must retain the upstream attention calculation."""

import unittest

import torch
from torch import nn

from drama_low_vram.codec import (
    _bounded_codec_transformers,
    _check_frame_budget,
)


class CodecBudgetTests(unittest.TestCase):
    def test_context_extends_past_training_block_without_changing_output(self):
        from fish_speech.models.dac import modded_dac

        config = modded_dac.ModelArgs(
            block_size=8,
            n_layer=1,
            n_head=2,
            n_local_heads=2,
            dim=128,
            intermediate_size=256,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
        )
        original = modded_dac.Transformer.__init__
        torch.manual_seed(610)
        with _bounded_codec_transformers(16):
            bounded = modded_dac.WindowLimitedTransformer(
                config, input_dim=128, window_size=8, causal=True
            ).eval()
        self.assertIs(modded_dac.Transformer.__init__, original)
        with _bounded_codec_transformers(32):
            reference = modded_dac.WindowLimitedTransformer(
                config, input_dim=128, window_size=8, causal=True
            ).eval()
        reference.load_state_dict(bounded.state_dict(), strict=True)
        self.assertEqual(bounded.freqs_cis.shape[0], 64)
        self.assertEqual(bounded.causal_mask.shape, (64, 64))
        x = torch.randn(1, 128, 48)
        with torch.inference_mode():
            torch.testing.assert_close(bounded(x), reference(x), rtol=0, atol=0)
        self.assertEqual(set(bounded.state_dict()), set(reference.state_dict()))
        self.assertNotIn("causal_mask", bounded.state_dict())
        self.assertNotIn("freqs_cis", bounded.state_dict())

    def test_constructor_is_restored_after_error(self):
        from fish_speech.models.dac import modded_dac

        original = modded_dac.Transformer.__init__
        with self.assertRaisesRegex(RuntimeError, "construction aborted"):
            with _bounded_codec_transformers(4):
                raise RuntimeError("construction aborted")
        self.assertIs(modded_dac.Transformer.__init__, original)

    def test_frame_limit_fails_before_codec_work(self):
        codec = nn.Module()
        codec._drama_low_vram_max_frames = 16
        _check_frame_budget(codec, 16)
        with self.assertRaisesRegex(ValueError, "reload with a larger"):
            _check_frame_budget(codec, 17)
