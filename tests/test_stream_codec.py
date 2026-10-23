"""Causal history, finite attention windows and retry-safe streaming commits."""

from types import SimpleNamespace

import pytest
import torch

from drama_low_vram.codec import _bounded_codec_transformers
from drama_low_vram.stream_codec import CausalCodecStream
from fish_speech.models.dac import modded_dac, rvq


def state():
    result = object.__new__(CausalCodecStream)
    result._dac, result._rvq = modded_dac, rvq
    result._state = {}
    result.latent_frames, result.decoded_frames, result._pending = 0, 0, 0
    result.closed, result.maximum = False, 64
    return result


@pytest.mark.parametrize("kind", ["conv", "dilated", "transpose", "upsample", "convnext"])
def test_streaming_convolutions_preserve_full_causal_context_at_irregular_boundaries(kind):
    torch.manual_seed(35)
    if kind == "conv":
        module = modded_dac.CausalConvNet(4, 6, 7)
    elif kind == "dilated":
        module = modded_dac.CausalConvNet(4, 6, 7, dilation=3)
    elif kind == "transpose":
        module = modded_dac.CausalTransConvNet(4, 6, 8, stride=4)
    elif kind == "upsample":
        module = rvq.CausalTransConvNet(4, 6, 2, stride=2)
    else:
        module = rvq.ConvNeXtBlock(4)
    module.eval()
    values = torch.randn(1, 4, 29)
    expected = module(values)
    streamed = state()
    parts = [
        streamed._apply(module, values[..., begin:end])
        for begin, end in ((0, 1), (1, 9), (9, 12), (12, 29))
    ]
    torch.testing.assert_close(torch.cat(parts, dim=-1), expected, rtol=1e-5, atol=2e-6)
    assert all(not value.is_cuda for value in streamed._state.values()
               if isinstance(value, torch.Tensor))


def test_transformer_keeps_per_layer_window_and_global_positions():
    torch.manual_seed(53)
    config = modded_dac.ModelArgs(
        dim=32, n_head=2, head_dim=16, n_local_heads=2,
        intermediate_size=64, n_layer=2, channels_first=True,
        dropout_rate=0.0, attn_dropout_rate=0.0)
    with _bounded_codec_transformers(64):
        module = modded_dac.WindowLimitedTransformer(
            config, input_dim=32, window_size=4, causal=True).eval()
    values = torch.randn(1, 32, 19)
    expected = module(values)
    streamed = state()
    actual = torch.cat([
        streamed._apply(module, values[..., begin:end])
        for begin, end in ((0, 2), (2, 9), (9, 10), (10, 19))
    ], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    assert streamed._state[(id(module), "position")] == 19
    for key, tensor in streamed._state.items():
        if key[1] in ("key", "value"):
            assert tensor.shape[2] == 3
    assert not module.use_kv_cache
    assert all(layer.attention.kv_cache is None for layer in module.layers)


def test_operation_failure_restores_immutable_state_and_can_be_retried():
    streamed = state()
    original = torch.arange(8)
    streamed._state["previous"] = original
    streamed.latent_frames, streamed.decoded_frames = 10, 10
    with pytest.raises(MemoryError):
        with streamed.transaction():
            streamed._save("previous", torch.zeros(8))
            streamed._save("extra", torch.ones(2))
            streamed.latent_frames, streamed._pending = 14, 4
            raise MemoryError("Simulated codec allocation failure")
    assert streamed._state == {"previous": original}
    assert (streamed.latent_frames, streamed.decoded_frames, streamed._pending) == (10, 10, 0)
    with streamed.transaction():
        streamed._save("previous", torch.ones(8))
    assert torch.equal(streamed._state["previous"], torch.ones(8))
    streamed.close()
    assert streamed.state_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        with streamed.transaction():
            pass


def test_pair_failure_rolls_back_already_committed_latent_stage():
    streamed = state()
    original = torch.arange(4)
    streamed._state["latent"] = original
    streamed.latent_frames = streamed.decoded_frames = 8
    with pytest.raises(MemoryError):
        with streamed.transaction():
            with streamed.transaction():
                streamed._save("latent", torch.ones(4))
                streamed.latent_frames, streamed._pending = 12, 4
            with streamed.transaction():
                streamed._save("waveform", torch.zeros(4))
                raise MemoryError("Waveform stage could not coexist")
    assert streamed._state == {"latent": original}
    assert (streamed.latent_frames, streamed.decoded_frames, streamed._pending) == (8, 8, 0)
