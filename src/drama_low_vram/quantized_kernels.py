"""Standalone quantized Drama3 GEMMs; no ComfyUI runtime dependency.

GPTQ W4A16 uses the published symmetric group-128 nibble layout.
INT8/W6A8 use regular Hadamard ConvRot and rowwise INT8 activations.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .nvidia_policy import uses_quantized_torch_fallback


@triton.jit
def _row_quant(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + i, i < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 127.
    q = libdevice.rint(x / scale)
    tl.store(Q + row * K + i, tl.minimum(127., tl.maximum(-127., q)).to(tl.int8), i < K)
    tl.store(S + row, scale)


_H = {}
_FLOAT8_VALUES = {}


def rotate_quantize(x):
    """Published regular H4 Kronecker basis (not the Sylvester Hadamard)."""
    key = x.device, x.dtype
    if key not in _H:
        h = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                          [1, -1, 1, 1], [-1, 1, 1, 1]], device=x.device, dtype=x.dtype)
        result = h
        for _ in range(3):
            result = torch.kron(result, h)
        _H[key] = result / 16.
    shape = x.shape
    rotated = torch.matmul(x.reshape(-1, 256), _H[key]).reshape(shape).contiguous()
    q = torch.empty_like(rotated, dtype=torch.int8)
    scale = torch.empty(shape[0], device=x.device, dtype=torch.float32)
    _row_quant[(shape[0],)](rotated, q, scale, K=shape[1], BLOCK=triton.next_power_of_2(shape[1]))
    return q, scale


def projection_torch(x, ids, weights, bank, scales, *, mode, shared=False, weighted=False):
    """Bounded fallback: reconstruct only a used expert, preserving its format.

    FLOAT8 scale bytes use an exact 256-entry lookup on Turing/Ampere. Expert weights
    remain packed in the checkpoint and source bank; no alternative model is
    loaded. FP32 products avoid requiring BF16/FP8 tensor-core instructions.
    """
    original_dtype = x.dtype
    if mode:
        key = x.device, x.dtype
        if key not in _H:
            h = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                              [1, -1, 1, 1], [-1, 1, 1, 1]], device=x.device, dtype=x.dtype)
            result = h
            for _ in range(3):
                result = torch.kron(result, h)
            _H[key] = result / 16.
        rotated = torch.matmul(x.reshape(-1, 256), _H[key]).reshape(x.shape).float()
        activation_scale = rotated.abs().amax(1).clamp_min(1e-12) / 127
        inputs = (rotated / activation_scale[:, None]).round().clamp(-127, 127)
    else:
        inputs = x.float()
    output = torch.empty((ids.numel(), bank.shape[1]), dtype=torch.float32, device=x.device)
    k = x.shape[1]
    groups = {}
    for row, expert in enumerate(ids.reshape(-1).tolist()):
        groups.setdefault(expert, []).append(row)
    for expert, rows in groups.items():
        source = bank[expert]
        auxiliary = scales[expert]
        if mode == 0:
            shifts = torch.arange(8, device=x.device) * 4
            code = ((source.long()[..., None] >> shifts) & 15).reshape(source.shape[0], k)
            matrix = ((code.float() - 8) * auxiliary.float().repeat_interleave(128, -1)).to(original_dtype).float()
        elif mode == 1:
            matrix = source.float() * auxiliary.float()
        else:
            raw = source.to(torch.uint8)
            low = raw[:, :k // 2]
            code = torch.stack((low & 15, low >> 4), -1).reshape(source.shape[0], k)
            if mode == 2:
                high = raw[:, k // 2:]
                high = ((high[..., None].long() >> (torch.arange(4, device=x.device) * 2)) & 3).reshape(source.shape[0], k)
                level = (code.long() | (high << 4)).float() - 32
            else:
                table = auxiliary[:16, k // 16 + 4:].contiguous().view(torch.float32).reshape(16)
                level = table[code.long()]
            if x.device not in _FLOAT8_VALUES:
                _FLOAT8_VALUES[x.device] = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(
                    torch.float8_e4m3fn).float().to(x.device)
            relative = _FLOAT8_VALUES[x.device][auxiliary[:, :k // 16].long()]
            channel = auxiliary[:, k // 16:k // 16 + 4].contiguous().view(torch.float32)
            matrix = (level * relative.repeat_interleave(16, -1)).round().clamp(-127, 127) * channel
        row_ids = torch.tensor(rows, dtype=torch.long, device=x.device)
        indices = torch.zeros_like(row_ids) if shared else row_ids
        values = F.linear(inputs.index_select(0, indices), matrix)
        if mode:
            values *= activation_scale.index_select(0, indices)[:, None]
        # Published projection boundary is BF16; represent it in the model's
        # compatible compute type afterward (FP16 on Turing).
        values = values.to(torch.bfloat16).float()
        if weighted:
            values *= weights.reshape(-1).index_select(0, row_ids)[:, None]
        output.index_copy_(0, row_ids, values)
    return output if weighted else output.to(original_dtype)


@triton.jit
def _quant_gemm(
    A, AS, B, BS, IDs, Weights, O,
    N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr,
    SHARED: tl.constexpr, WEIGHTED: tl.constexpr,
    BE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SE: tl.constexpr, SN: tl.constexpr, SK: tl.constexpr,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 128,
):
    pair, tile = tl.program_id(0), tl.program_id(1)
    expert = tl.load(IDs + pair).to(tl.int64)
    m = tl.arange(0, 16)
    n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    row = 0 if SHARED else pair
    if MODE == 0:
        acc = tl.zeros((16, BLOCK_N), tl.float32)
    else:
        acc = tl.zeros((16, BLOCK_N), tl.int32)
    for block in range(K // BLOCK_K):
        ki = block * BLOCK_K + k
        a = tl.load(A + row * K + ki[None, :] + tl.zeros((16, 1), tl.int32),
                    m[:, None] == 0, 0)
        if MODE == 0:
            # [expert,N,K/8] INT32, little-endian nibbles, implicit zp8.
            word = tl.load(B + expert * BE + n[None, :] * BN + (ki[:, None] // 8) * BK,
                           n[None, :] < N, 0)
            code = (word >> ((ki[:, None] % 8) * 4)) & 15
            scale = tl.load(BS + expert * SE + n[None, :] * SN + (ki[:, None] // 128) * SK,
                            n[None, :] < N, 0).to(tl.float32)
            b = ((code.to(tl.float32) - 8.) * scale).to(a.dtype)
            acc += tl.dot(a, b)
        elif MODE == 1:
            b = tl.load(B + expert * BE + n[None, :] * BN + ki[:, None] * BK,
                        n[None, :] < N, 0)
            acc += tl.dot(a, b)
        else:
            # Uniform six-bit codes: low-nibble plane followed by top-two bits.
            low = tl.load(B + expert * BE + n[None, :] * BN + (ki[:, None] // 2) * BK,
                          n[None, :] < N, 0).to(tl.int32) & 255
            low = (low >> (4 * (ki[:, None] % 2))) & 15
            if MODE == 2:
                high = tl.load(B + expert * BE + n[None, :] * BN
                               + (K // 2 + ki[:, None] // 4) * BK,
                               n[None, :] < N, 0).to(tl.int32) & 255
                high = (high >> (2 * (ki[:, None] % 4))) & 3
                code = low | (high << 4)
                level = code.to(tl.float32) - 32.
            else:
                codebook = BS + expert * SE + low.to(tl.int64) * SN + K // 16 + 4
                level = tl.load(tl.cast(codebook, tl.pointer_type(tl.float32)),
                                n[None, :] < N, 0)
            raw = tl.load(BS + expert * SE + n[None, :] * SN + (ki[:, None] // 16) * SK,
                          n[None, :] < N, 0)
            scale = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            b = libdevice.rint(level * scale)
            b = tl.minimum(127., tl.maximum(-127., b)).to(tl.int8)
            acc += tl.dot(a, b)
    result = acc.to(tl.float32)
    if MODE != 0:
        sa = tl.load(AS + row)
        if MODE == 1:
            sb = tl.load(BS + expert * SE + n * SN, n < N, 0).to(tl.float32)
        else:
            ptr = BS + expert * SE + n * SN + K // 16
            sb = tl.load(tl.cast(ptr, tl.pointer_type(tl.float32)), n < N, 0)
        result = result * sa * sb[None, :]
    if WEIGHTED:
        # The published INT8 GEMM returns BF16 before route weighting.
        # GPTQ also emits BF16 expert output; preserve this boundary.
        result = result.to(tl.bfloat16).to(tl.float32)
        result = result * tl.load(Weights + pair).to(tl.float32)
    tl.store(O + pair * N + n[None, :] + tl.zeros((16, 1), tl.int32),
             result.to(O.dtype.element_ty), (m[:, None] == 0) & (n[None, :] < N))


def projection(x, ids, weights, bank, scales, *, mode, shared=False, weighted=False,
               tile=(64, 128)):
    if x.device.type != "cuda" or uses_quantized_torch_fallback(torch.cuda.get_device_capability(x.device), mode):
        return projection_torch(x, ids, weights, bank, scales, mode=mode, shared=shared, weighted=weighted)
    count = ids.numel()
    n = bank.shape[1]
    k = x.shape[1]
    if mode != 0:
        x, activation_scale = rotate_quantize(x)
    else:
        activation_scale = x  # Unused pointer.
    out = torch.empty((count, n), device=x.device,
                      dtype=torch.float32 if weighted else torch.bfloat16)
    block_n, block_k = tile
    _quant_gemm[(count, triton.cdiv(n, block_n))](
        x, activation_scale, bank, scales, ids.reshape(-1).contiguous(),
        weights.reshape(-1).contiguous(), out,
        N=n, K=k, MODE=mode, SHARED=shared, WEIGHTED=weighted,
        BE=bank.stride(0), BN=bank.stride(1), BK=bank.stride(2),
        SE=scales.stride(0), SN=scales.stride(1), SK=scales.stride(2),
        BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=2)
    return out


class QuantizedExpertBackend:
    def __init__(self, mode):
        self.mode = mode
        self.tuned_tiles = True

    def forward(self, x, ids, weights, staged):
        tokens, top_k = ids.shape
        if tokens == 1:
            partials = self._pairs(x, ids, weights, staged, shared=True)
        else:
            partials = self._pairs(
                x.repeat_interleave(top_k, dim=0), ids, weights, staged, shared=False)
        return partials.view(tokens, top_k, x.shape[-1]).sum(1).to(x.dtype)

    def _pairs(self, x, ids, weights, staged, *, shared=False):
        up_tile = down_tile = (64, 128)
        if self.tuned_tiles and torch.cuda.get_device_capability(x.device) == (8, 9):
            if self.mode == 0:
                up_tile = (32, 256)
                down_tile = (32, 128 if ids.numel() <= 8 else 256)
            elif self.mode in (2, 3) and ids.numel() <= 8:
                up_tile = down_tile = (32, 128)
        up = projection(x, ids, weights, staged.w13_weight, staged.w13_scale_inv,
                        mode=self.mode, shared=shared, tile=up_tile)
        gate, value = up.chunk(2, dim=-1)
        activated = F.silu(gate) * value
        return projection(
            activated, ids, weights, staged.w2_weight, staged.w2_scale_inv,
            mode=self.mode, weighted=True, tile=down_tile)

    def forward_pairs(self, x, ids, weights, staged):
        return self._pairs(x, ids, weights, staged)
