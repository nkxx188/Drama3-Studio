"""Packed quantized math against explicit reconstruction, without model downloads."""

import pytest
import torch
from drama_low_vram.quantized_kernels import projection, projection_torch, rotate_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def gptq_pack(codes):
    n, k = codes.shape
    words = torch.zeros(n, k // 8, dtype=torch.int64, device=codes.device)
    for shift in range(8):
        words |= codes[:, shift::8].long() << (4 * shift)
    return words.to(torch.int32)


def six_pack(codes):
    n, k = codes.shape
    low = (codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)
    high = torch.zeros(n, k // 4, dtype=torch.int64, device=codes.device)
    for shift in range(4):
        high |= (codes[:, shift::4].long() >> 4) << (2 * shift)
    return torch.cat((low.to(torch.uint8), high.to(torch.uint8)), dim=-1).view(torch.int8)


@torch.inference_mode()
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tile", [(64, 128), (32, 128), (32, 256)])
@pytest.mark.parametrize("compat", [False, True])
def test_quantized_projection_matches_explicit_grid_and_bf16_boundaries(mode, weighted, tile, compat):
    torch.manual_seed(71)
    n, k, pairs = 128, 256, 3
    x = torch.randn(pairs, k, device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([0.2, 0.6, 0.3], device="cuda")
    ids = torch.tensor([1, 0, 1], device="cuda")
    if mode == 0:
        codes = torch.randint(0, 16, (2, n, k), device="cuda")
        bank = torch.stack([gptq_pack(value) for value in codes])
        scales = (torch.rand(2, n, k // 128, device="cuda") * 0.03).half()
        explicit = ((codes.float() - 8) * scales.repeat_interleave(128, dim=-1)).bfloat16()
        inputs = x.float()
    elif mode == 1:
        bank = torch.randint(-127, 128, (2, n, k), device="cuda", dtype=torch.int8)
        scales = torch.rand(2, n, 1, device="cuda") * 0.02
        explicit = bank.float() * scales
        q, s = rotate_quantize(x)
        inputs = q.float() * s[:, None]
    else:
        codes = torch.randint(0, 16 if mode == 3 else 64, (2, n, k), device="cuda")
        if mode == 3:
            bank = (((codes[..., 0::2] & 15)
                     | ((codes[..., 1::2] & 15) << 4)).to(torch.uint8).view(torch.int8))
        else:
            bank = torch.stack([six_pack(value) for value in codes])
        relative = (torch.rand(2, n, k // 16, device="cuda") + 1).to(torch.float8_e4m3fn)
        channel = torch.rand(2, n, 1, device="cuda") * 0.02
        if mode == 3:
            tables = torch.linspace(-12, 13, 16, device="cuda").repeat(2, 1).contiguous()
            scales = torch.zeros(2, n, k // 16 + 8, device="cuda", dtype=torch.uint8)
            scales[..., :k // 16] = relative.view(torch.uint8)
            scales[..., k // 16:k // 16 + 4] = channel.view(torch.uint8)
            scales[:, :16, k // 16 + 4:] = tables.view(torch.uint8).reshape(2, 16, 4)
            levels = torch.stack([tables[i][codes[i]] for i in range(2)])
        else:
            scales = torch.cat((relative.view(torch.uint8), channel.view(torch.uint8)), dim=-1)
            levels = codes.float() - 32
        grid = levels * relative.float().repeat_interleave(16, dim=-1)
        explicit = grid.round().clamp(-127, 127) * channel
        q, s = rotate_quantize(x)
        inputs = q.float() * s[:, None]
    expected = torch.stack([
        inputs[index] @ explicit[expert].float().T for index, expert in enumerate(ids.tolist())])
    expected = expected.bfloat16()
    if weighted:
        expected = expected.float() * weights[:, None]
    actual = (projection_torch(x, ids, weights, bank, scales, mode=mode, weighted=weighted) if compat else
              projection(x, ids, weights, bank, scales, mode=mode, weighted=weighted, tile=tile))
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.02)
    # Routed weighting must remain FP32, rather than a low precision sum.
    assert actual.dtype == (torch.float32 if weighted else torch.bfloat16)


@torch.inference_mode()
def test_compatible_projection_accepts_fp16_turing_compute_without_bf16_gemm():
    torch.manual_seed(14)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16)
    ids = torch.tensor([0, 1, 0], device="cuda")
    weights = torch.tensor([0.2, 0.3, 0.4], device="cuda")
    bank = torch.randint(-127, 128, (2, 64, 256), dtype=torch.int8, device="cuda")
    scales = torch.ones((2, 64, 1), device="cuda") * 0.02
    q, activation_scale = rotate_quantize(x)
    expected = torch.stack([
        (q[row].float() @ (bank[expert].float() * scales[expert]).T * activation_scale[row])
        for row, expert in enumerate(ids.tolist())
    ]).bfloat16().half()
    actual = projection_torch(x, ids, weights, bank, scales, mode=1)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
