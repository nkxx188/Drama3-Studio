"""Run official DAC decoder steps without retaining redundant block inputs."""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn


def _flat_forward(self, x):
    if self.training or torch.is_grad_enabled():
        return type(self).forward(self, x)
    for step in self._drama_decoder_steps:
        x = step(x)
    return x


def flatten_decoder_calls(codec):
    """Preserve every leaf operation and all residual units in their original order."""
    if (type(codec).__module__ != "fish_speech.models.dac.modded_dac"
            or type(codec).__name__ != "DAC"):
        return False
    from fish_speech.models.dac.modded_dac import DAC, Decoder, DecoderBlock

    if type(codec) is not DAC or type(codec.decoder) is not Decoder:
        return False

    def steps(module):
        if type(module) in (nn.Sequential, DecoderBlock):
            if (module._forward_hooks or module._forward_pre_hooks
                    or module._backward_hooks or module._backward_pre_hooks):
                raise ValueError("Decoder container hooks require its original call")
            children = module.block if type(module) is DecoderBlock else module
            for child in children:
                yield from steps(child)
        else:
            yield module

    try:
        sequence = tuple(steps(codec.decoder.model))
    except ValueError:
        return False
    codec.decoder._drama_decoder_steps = sequence
    codec.decoder.forward = MethodType(_flat_forward, codec.decoder)
    codec._drama_flat_decoder = True
    return True


def decoder_workspace_bytes(codec, frames):
    """Bound the audited BF16 1536-channel decoder's live activation tensors."""
    if not getattr(codec, "_drama_flat_decoder", False) or not 1 <= frames <= 650:
        return None
    from fish_speech.models.dac.modded_dac import CausalTransConvNet, DecoderBlock

    blocks = [step for step in codec.decoder.model if type(step) is DecoderBlock]
    expected = ((1536, 768, 8), (768, 384, 8), (384, 192, 4), (192, 96, 2))
    if len(blocks) != len(expected):
        return None
    for block, (inputs, outputs, stride) in zip(blocks, expected):
        wrapper = block.block[1]
        if type(wrapper) is not CausalTransConvNet:
            return None
        conv = wrapper.conv
        if (type(conv) is not nn.ConvTranspose1d or conv.in_channels != inputs
                or conv.out_channels != outputs or conv.stride != (stride,)
                or conv.kernel_size != (stride * 2,) or conv.dilation != (1,)
                or conv.padding != (0,) or conv.output_padding != (0,) or conv.groups != 1
                or conv.weight.dtype != torch.bfloat16):
            return None
    # Six largest BF16 activation planes (96 channels x 2048 samples/frame),
    # plus 64MiB for fixed workspaces and alignment. Audited at 16/375/650 frames.
    return 64 * 1024**2 + frames * 6 * 96 * 2048 * 2
