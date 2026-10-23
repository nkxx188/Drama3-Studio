"""Batch-one Drama MoE: route slots directly, with the official block arithmetic."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def silu_table_kernel(output, BLOCK: tl.constexpr):
    bits = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(output + bits, gate * tl.sigmoid(gate))


def build_gpu_silu_table(device="cuda:0"):
    """Exact GPU SiLU for every BF16 bit pattern; 256KiB CPU research table."""
    table = torch.empty(65536, dtype=torch.float32, device=device)
    silu_table_kernel[(65536 // 128,)](table, BLOCK=128, num_warps=1)
    return table.cpu()


@triton.jit
def decode_gemm(
    a_ptr, b_ptr, out_ptr, a_scale_ptr, b_scale_ptr, ids_ptr, weights_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_be: tl.constexpr, stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    stride_bse: tl.constexpr, stride_bsn: tl.constexpr, stride_bsk: tl.constexpr,
    stride_asm: tl.constexpr, SHARED_INPUT: tl.constexpr,
    ROUTE_WEIGHT: tl.constexpr,
):
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    m = tl.arange(0, 16)
    n = tile * 128 + tl.arange(0, 128)
    k = tl.arange(0, 128)
    row = 0 if SHARED_INPUT else pair
    a_ptrs = a_ptr + row * K + k[None, :] + tl.zeros((16, 1), tl.int32)
    b_ptrs = b_ptr + expert * stride_be + n[None, :] * stride_bn + k[:, None] * stride_bk
    acc = tl.zeros((16, 128), dtype=tl.float32)
    for block in range(K // 128):
        a = tl.load(a_ptrs, mask=m[:, None] == 0, other=0.)
        b = tl.load(b_ptrs, mask=n[None, :] < N, other=0.)
        a_scale = tl.load(
            a_scale_ptr + row * stride_asm + block + tl.zeros((16,), tl.int32),
            mask=m == 0, other=0.)
        b_scale = tl.load(
            b_scale_ptr + expert * stride_bse + (n // 128) * stride_bsn + block * stride_bsk)
        acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += 128
        b_ptrs += 128 * stride_bk
    if ROUTE_WEIGHT:
        acc = acc * tl.load(weights_ptr + pair).to(tl.float32)
    output = out_ptr + pair * N + n[None, :] + tl.zeros((16, 1), tl.int32)
    tl.store(output, acc.to(out_ptr.dtype.element_ty),
             mask=(m[:, None] == 0) & (n[None, :] < N))


@triton.jit
def decode_gemm_sources(
    A, Cold, Hot, O, AS, ColdS, HotS, IDs, Sources, Weights,
    N: tl.constexpr, K: tl.constexpr,
    SBE: tl.constexpr, SBN: tl.constexpr, SBK: tl.constexpr,
    SSE: tl.constexpr, SSN: tl.constexpr, SSK: tl.constexpr,
    SAS: tl.constexpr, SHARED: tl.constexpr, WEIGHTED: tl.constexpr,
):
    pair, tile = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDs + pair).to(tl.int64)
    hot = tl.load(Sources + pair) != 0
    base, scales = tl.where(hot, Hot, Cold), tl.where(hot, HotS, ColdS)
    m, n, k = tl.arange(0, 16), tile * 128 + tl.arange(0, 128), tl.arange(0, 128)
    row = 0 if SHARED else pair
    ap = A + row * K + k[None, :] + tl.zeros((16, 1), tl.int32)
    bp = base + expert * SBE + n[None, :] * SBN + k[:, None] * SBK
    acc = tl.zeros((16, 128), tl.float32)
    for block in range(K // 128):
        a = tl.load(ap, m[:, None] == 0, 0.)
        b = tl.load(bp, n[None, :] < N, 0.)
        sa = tl.load(AS + row * SAS + block + tl.zeros((16,), tl.int32), m == 0, 0.)
        sb = tl.load(scales + expert * SSE + (n // 128) * SSN + block * SSK)
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
        ap += 128
        bp += 128 * SBK
    if WEIGHTED:
        acc = acc * tl.load(Weights + pair).to(tl.float32)
    tl.store(O + pair * N + n[None, :] + tl.zeros((16, 1), tl.int32),
             acc.to(O.dtype.element_ty), (m[:, None] == 0) & (n[None, :] < N))


def quantized_pairs_sources(kernel, q, scales, ids, sources, weights, cold, hot):
    """Research path: borrow resident weights directly; avoid mixed-bank copies."""
    count, hidden = ids.numel(), q.shape[1]
    gu = torch.empty((count, cold.w13_weight.shape[1]), device=q.device, dtype=torch.bfloat16)
    down = torch.empty((count, hidden), device=q.device, dtype=torch.float32)
    for name, output, weighted in (("w13", gu, False), ("w2", down, True)):
        if weighted:
            q, scales = kernel.quantize_groups(kernel.silu_mul_quant_kernel, gu, gu.shape[1] // 2)
        bank, scale = getattr(cold, name + "_weight"), getattr(cold, name + "_scale_inv")
        other, other_scale = getattr(hot, name + "_weight"), getattr(hot, name + "_scale_inv")
        if bank.stride() != other.stride() or scale.stride() != other_scale.stride():
            raise ValueError("Direct source banks require identical expert geometry")
        decode_gemm_sources[(count, triton.cdiv(bank.shape[1], 128))](
            q, bank, other, output, scales, scale, other_scale, ids, sources, weights,
            N=bank.shape[1], K=bank.shape[2],
            SBE=bank.stride(0), SBN=bank.stride(1), SBK=bank.stride(2),
            SSE=scale.stride(0), SSN=scale.stride(1), SSK=scale.stride(2),
            SAS=scales.stride(0), SHARED=not weighted, WEIGHTED=weighted,
            num_warps=4, num_stages=3)
    return down


def forward(kernel, x, ids, weights, staged):
    """One input row, up to eight route slots, original FP8/scales and reduction."""
    hidden = x.shape[1]
    q, scales = kernel.quantize_groups(kernel.quant_kernel, x.contiguous(), hidden)
    down = quantized_pairs(kernel, q, scales, ids, weights, staged, dtype=x.dtype)
    return down.sum(0, keepdim=True).to(x.dtype)


def quantized_pairs(kernel, q, scales, ids, weights, staged, *, dtype=torch.bfloat16):
    """Ordered FP32 partials, also usable with a shared CPU/GPU FP8 input."""
    count, hidden = ids.numel(), q.shape[1]
    ids = ids.reshape(-1).contiguous()
    gu = torch.empty(
        (count, staged.w13_weight.shape[1]), device=q.device, dtype=dtype)
    down = torch.empty((count, hidden), device=q.device, dtype=torch.float32)
    for source, source_scales, bank, bank_scales, output, shared, weighted in (
        (q, scales, staged.w13_weight, staged.w13_scale_inv, gu, q.shape[0] == 1, False),
        (None, None, staged.w2_weight, staged.w2_scale_inv, down, False, True),
    ):
        if output is down:
            source, source_scales = kernel.quantize_groups(
                kernel.silu_mul_quant_kernel, gu, gu.shape[1] // 2)
        decode_gemm[(count, triton.cdiv(bank.shape[1], 128))](
            source, bank, output, source_scales, bank_scales, ids,
            weights.reshape(-1).contiguous(),
            N=bank.shape[1], K=bank.shape[2],
            stride_be=bank.stride(0), stride_bn=bank.stride(1), stride_bk=bank.stride(2),
            stride_bse=bank_scales.stride(0), stride_bsn=bank_scales.stride(1),
            stride_bsk=bank_scales.stride(2),
            stride_asm=source_scales.stride(0),
            SHARED_INPUT=shared, ROUTE_WEIGHT=weighted,
            num_warps=4, num_stages=3,
        )
    return down
