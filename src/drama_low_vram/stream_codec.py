"""Causal state for the unchanged official codec, independent of weight placement.

Each operation commits CPU state only after success. A failed allocation rolls
back cheaply because saved state tensors are immutable; the engine may retry
the same operation after parking its backbone.
"""

from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


class CausalCodecStream:
    def __init__(self, codec, *, max_frames=1024):
        from fish_speech.models.dac import modded_dac, rvq

        if type(codec) is not modded_dac.DAC or type(codec.quantizer) is not rvq.DownsampleResidualVectorQuantize:
            raise ValueError("Streaming requires the audited official DAC codec")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("Streaming codec frame limit must be positive")
        self.codec, self.maximum = codec, max_frames
        self._dac, self._rvq = modded_dac, rvq
        self._state = {}
        self.latent_frames, self.decoded_frames, self._pending = 0, 0, 0
        self.closed = False

    @property
    def state_bytes(self):
        return sum(value.numel() * value.element_size()
                   for value in self._state.values() if isinstance(value, torch.Tensor))

    @contextmanager
    def transaction(self):
        """Commit a stage or a complete latent/waveform pair only on success."""
        if self.closed:
            raise RuntimeError("This codec stream is closed")
        previous = self._state.copy()
        counts = self.latent_frames, self.decoded_frames, self._pending
        try:
            yield
        except BaseException:
            self._state = previous
            self.latent_frames, self.decoded_frames, self._pending = counts
            raise

    def _save(self, key, value):
        self._state[key] = value.detach().to(device="cpu", copy=True).contiguous()

    def _previous(self, key, current, count):
        value = self._state.get(key)
        if value is None:
            return current.new_zeros((*current.shape[:-1], count))
        return value.to(device=current.device, dtype=current.dtype)

    def _conv(self, module, x):
        if module.stride != 1:
            raise ValueError("Only decoder stride-one causal convolutions are audited")
        key, count = (id(module), "conv"), module.padding
        previous = self._previous(key, x, count)
        joined = torch.cat((previous, x), dim=-1) if count else x
        y = module.conv(joined).contiguous()
        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError("Causal convolution changed the streaming time axis")
        if count:
            self._save(key, joined[..., -count:])
        return y

    def _transpose_conv(self, module, x):
        if module.conv.dilation != (1,) or module.conv.padding != (0,):
            raise ValueError("Unaudited streaming transposed convolution geometry")
        count = (module.kernel_size - 1) // module.stride
        key = (id(module), "transpose")
        previous = self._previous(key, x, count)
        joined = torch.cat((previous, x), dim=-1) if count else x
        # Original padding/cropping, including bias, stays in its own forward.
        y = module(joined)[..., count * module.stride:]
        if y.shape[-1] != x.shape[-1] * module.stride:
            raise RuntimeError("Causal transposed convolution changed output length")
        if count:
            self._save(key, joined[..., -count:])
        return y.contiguous()

    def _attention(self, module, x, positions, frequencies, window):
        batch, length, _ = x.shape
        width = module.n_local_heads * module.head_dim
        q, k, v = module.wqkv(x).split((width, width, width), dim=-1)
        q = q.view(batch, length, module.n_head, module.head_dim)
        k = k.view(batch, length, module.n_local_heads, module.head_dim)
        v = v.view(batch, length, module.n_local_heads, module.head_dim)
        q = self._dac.apply_rotary_emb(q, frequencies)
        k = self._dac.apply_rotary_emb(k, frequencies)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        old_k = self._state.get((id(module), "key"))
        old_v = self._state.get((id(module), "value"))
        past = 0 if old_k is None else old_k.shape[2]
        if past:
            k = torch.cat((old_k.to(k.device), k), dim=2)
            v = torch.cat((old_v.to(v.device), v), dim=2)
        keep = min(window - 1, k.shape[2])
        if keep:
            self._save((id(module), "key"), k[:, :, -keep:])
            self._save((id(module), "value"), v[:, :, -keep:])
        groups = module.n_head // module.n_local_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        columns = torch.arange(-past, length, device=x.device) + positions[0]
        mask = (columns[None, :] <= positions[:, None])
        mask &= columns[None, :] >= positions[:, None] - window + 1
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(batch, length, module.head_dim * module.n_head)
        return module.wo(y)

    def _transformer(self, module, x):
        if (not module.causal or module.config.pos_embed_type != "rope"
                or not isinstance(module.look_ahead_conv, nn.Identity)
                or module.use_kv_cache):
            raise ValueError("Unaudited streaming codec transformer configuration")
        if module.channels_first:
            x = x.transpose(1, 2)
        x = module.look_ahead_conv(module.input_proj(x))
        key = (id(module), "position")
        start = self._state.get(key, 0)
        length = x.shape[1]
        if start + length > module.freqs_cis.shape[0]:
            raise ValueError("Codec stream exceeds its positional buffer")
        positions = torch.arange(start, start + length, device=x.device)
        frequencies = module.freqs_cis[positions]
        window = module.window_size or self.maximum * 4
        for layer in module.layers:
            attended = self._attention(
                layer.attention, layer.attention_norm(x), positions, frequencies, window)
            h = x + layer.attention_layer_scale(attended)
            x = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
        x = module.output_proj(module.norm(x))
        self._state[key] = start + length
        return x.transpose(1, 2) if module.channels_first else x

    def _apply(self, module, x):
        if type(module) in (self._dac.CausalConvNet, self._rvq.CausalConvNet):
            return self._conv(module, x)
        if type(module) in (self._dac.CausalTransConvNet, self._rvq.CausalTransConvNet):
            return self._transpose_conv(module, x)
        if type(module) is self._dac.WindowLimitedTransformer:
            return self._transformer(module, x)
        if type(module) is nn.Sequential:
            for child in module:
                x = self._apply(child, x)
            return x
        if type(module) is self._dac.Decoder:
            return self._apply(module.model, x)
        if type(module) is self._dac.DecoderBlock:
            return self._apply(module.block, x)
        if type(module) is self._dac.ResidualUnit:
            if not module.causal:
                raise ValueError("A noncausal residual unit cannot stream")
            return x + self._apply(module.block, x)
        if type(module) is self._rvq.ConvNeXtBlock:
            hidden = self._apply(module.dwconv, x).permute(0, 2, 1)
            hidden = module.pwconv2(module.act(module.pwconv1(module.norm(hidden))))
            if module.gamma is not None:
                hidden = module.gamma * hidden
            return x + hidden.permute(0, 2, 1)
        if isinstance(module, (nn.Identity, nn.Linear, nn.LayerNorm, nn.Tanh, self._dac.Snake1d)):
            return module(x)
        raise ValueError(f"Unaudited codec stream module: {type(module).__name__}")

    @torch.inference_mode()
    def decode_latents(self, codes):
        with self.transaction():
            quantizer = self.codec.quantizer
            if (self._pending or codes.ndim != 2 or codes.shape[1] < 1
                    or codes.shape[0] != 1 + quantizer.quantizer.n_codebooks
                    or codes.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8)
                    or int(codes.min()) < 0
                    or self.latent_frames + codes.shape[1] > self.maximum):
                raise ValueError("Invalid streaming codes or unfinished waveform stage")
            device = next(quantizer.semantic_quantizer.parameters()).device
            indices = codes.to(device=device, dtype=torch.long, copy=True)[None]
            indices[:, 0].clamp_(max=quantizer.semantic_quantizer.codebook_size - 1)
            indices[:, 1:].clamp_(max=quantizer.quantizer.codebook_size - 1)
            semantic = quantizer.semantic_quantizer.from_codes(indices[:, :1])[0]
            residual = quantizer.quantizer.from_codes(indices[:, 1:])[0]
            latent = self._apply(quantizer.post_module, semantic + residual)
            latent = self._apply(quantizer.upsample, latent)
            self._pending = codes.shape[1]
            self.latent_frames += codes.shape[1]
            return latent

    @torch.inference_mode()
    def decode_waveform(self, latents):
        with self.transaction():
            if not self._pending:
                raise ValueError("Decode codec latents before their waveform")
            audio = self._apply(self.codec.decoder, latents)[0, 0].cpu()
            if audio.numel() != self._pending * 2048 or not bool(torch.isfinite(audio).all()):
                raise RuntimeError("Invalid streamed waveform")
            self.decoded_frames += self._pending
            self._pending = 0
            return audio

    def close(self):
        self._state.clear()
        self.closed = True
