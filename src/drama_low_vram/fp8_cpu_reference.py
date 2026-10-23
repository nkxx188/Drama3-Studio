"""Explicit batch-one W8A8 reference matching the official blockwise contract.

This is a slow validation implementation, not an inference backend. Accumulation
and exp implementation can round differently from GPU tensor cores.
"""

from __future__ import annotations

import torch

from .offload import ExpertStore, FP8_DTYPE


def quantize_fp8_blocks(values):
    if values.shape[-1] % 128 or not bool(torch.isfinite(values).all()):
        raise ValueError("Expected finite whole 128-value activation groups")
    blocks = values.float().reshape(-1, 128)
    scales = blocks.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448.0)
    raw = (blocks / scales[:, None]).clamp(-448, 448).to(FP8_DTYPE)
    return raw.reshape(values.shape), scales


def block_gemv(weight, scale, quantized, activation_scales):
    rows, columns = weight.shape
    blocks = weight.float().reshape(rows, columns // 128, 128)
    dot = (blocks * quantized.float().reshape(1, columns // 128, 128)).sum(-1)
    block_scales = scale.repeat_interleave(128, 0)
    acc = torch.zeros(rows)
    for k in range(columns // 128):
        scaled = dot[:, k] * activation_scales[k]
        # A float64 multiply/add rounds once to FP32, emulating the outer
        # fma.rn.f32 in the current GPU kernel for finite normal values.
        acc = (scaled.double() * block_scales[:, k].double() + acc.double()).float()
    return acc


@torch.inference_mode()
def expert_pairs(store: ExpertStore, x, ids, weights):
    """Ordered weighted FP32 contributions, before router reduction."""
    store.validate()
    if x.numel() != store.hidden or x.dtype != torch.bfloat16 or x.device.type != "cpu":
        raise ValueError("Reference expects one BF16 CPU hidden row")
    q, scales = quantize_fp8_blocks(x.reshape(-1))
    partials = []
    for expert, route_weight in zip(ids.tolist(), weights.tolist(), strict=True):
        gu = block_gemv(store.w13_weight[expert], store.w13_scale_inv[expert], q, scales)
        gate, up = gu.to(torch.bfloat16).float().chunk(2)
        hidden = (gate * torch.sigmoid(gate)) * up
        hq, hs = quantize_fp8_blocks(hidden)
        down = block_gemv(store.w2_weight[expert], store.w2_scale_inv[expert], hq, hs)
        partials.append(down * route_weight)
    return torch.stack(partials)
