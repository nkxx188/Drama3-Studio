"""Decoder lifetime changes preserve leaf math, hooks, weights and training."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from drama_low_vram.codec_decoder import (
    decoder_workspace_bytes, flatten_decoder_calls,
)
from fish_speech.models.dac.modded_dac import (
    CausalTransConvNet, DAC, Decoder, DecoderBlock,
)


def container(kind):
    value = object.__new__(kind)
    nn.Module.__init__(value)
    return value


def tiny_codec():
    codec, decoder, block = (container(kind) for kind in (DAC, Decoder, DecoderBlock))
    block.block = nn.Sequential(nn.Conv1d(2, 2, 3, padding=1), nn.SiLU())
    decoder.model = nn.Sequential(nn.Conv1d(2, 2, 1), block, nn.Tanh())
    codec.decoder = decoder
    return codec.eval()


def test_flat_decoder_preserves_outputs_weights_leaf_calls_and_gradients():
    codec = tiny_codec()
    inputs = torch.randn(1, 2, 29)
    weights = dict(codec.state_dict())
    called = []
    handle = codec.decoder.model[1].block[0].register_forward_hook(
        lambda *_: called.append(True))
    with torch.inference_mode():
        expected = codec.decoder(inputs)
        assert flatten_decoder_calls(codec)
        actual = codec.decoder(inputs)
    assert torch.equal(expected, actual)
    assert len(called) == 2
    assert weights.keys() == codec.state_dict().keys()
    assert all(weights[key].data_ptr() == value.data_ptr()
               for key, value in codec.state_dict().items())
    handle.remove()
    differentiable = inputs.clone().requires_grad_()
    codec.decoder(differentiable).sum().backward()
    assert differentiable.grad is not None


def test_container_hooks_and_unknown_codec_types_keep_the_original_path():
    codec = tiny_codec()
    original = codec.decoder.forward
    codec.decoder.model[1].register_forward_hook(lambda *_: None)
    assert not flatten_decoder_calls(codec)
    assert codec.decoder.forward == original
    assert not flatten_decoder_calls(SimpleNamespace(decoder=codec.decoder))
    assert decoder_workspace_bytes(codec, 375) is None


@pytest.mark.parametrize("frames", [16, 375, 650])
def test_workspace_bound_is_only_for_the_verified_bf16_decoder_layout(frames):
    codec = tiny_codec()
    blocks = []
    with torch.device("meta"):
        for inputs, outputs, stride in (
            (1536, 768, 8), (768, 384, 8), (384, 192, 4), (192, 96, 2)
        ):
            block = container(DecoderBlock)
            block.block = nn.Sequential(
                nn.Identity(),
                CausalTransConvNet(inputs, outputs, stride * 2, stride=stride).to(
                    dtype=torch.bfloat16))
            blocks.append(block)
    codec.decoder.model = nn.Sequential(*blocks)
    assert flatten_decoder_calls(codec)
    budget = decoder_workspace_bytes(codec, frames)
    measured = {16: 39739904, 375: 885543424, 650: 1534624768}[frames]
    assert budget > measured + 60 * 1024**2
    assert decoder_workspace_bytes(codec, 651) is None
    codec.decoder.model[0].block[1].conv.weight = nn.Parameter(
        codec.decoder.model[0].block[1].conv.weight.float())
    assert decoder_workspace_bytes(codec, frames) is None
