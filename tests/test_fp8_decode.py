"""Direct route-slot decoding must preserve the official FP8 arithmetic."""

import pytest
import torch

from drama_low_vram.offload import StagedExperts, TritonExpertBackend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hidden,intermediate", [(128, 128), (256, 256), (2048, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_direct_decode_matches_official_with_duplicate_and_permuted_routes(
    hidden, intermediate, dtype,
):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("SM89 FP8 kernel required")
    generator = torch.Generator().manual_seed(9274)
    experts = 8
    bank = StagedExperts(
        w13_weight=(torch.randn(experts, 2 * intermediate, hidden, generator=generator) * 50)
        .clamp(-448, 448).to(torch.float8_e4m3fn).cuda(),
        w13_scale_inv=(torch.rand(experts, 2 * intermediate // 128, hidden // 128, generator=generator)
                       * 0.02 + 0.001).cuda(),
        w2_weight=(torch.randn(experts, hidden, intermediate, generator=generator) * 50)
        .clamp(-448, 448).to(torch.float8_e4m3fn).cuda(),
        w2_scale_inv=(torch.rand(experts, hidden // 128, intermediate // 128, generator=generator)
                      * 0.02 + 0.001).cuda(),
    )
    original = TritonExpertBackend(decode_specialization=False)
    candidate = TritonExpertBackend(decode_specialization=True)
    for route in ([6], [7, 1], [3, 1, 3, 0, 2, 7, 1, 5], list(reversed(range(8)))):
        x = torch.randn(1, hidden, generator=generator).to(dtype).cuda()
        ids = torch.tensor([route], device=x.device, dtype=torch.int64)
        weights = torch.softmax(torch.randn(1, len(route), generator=generator), -1).to(dtype).cuda()
        expected = original.forward(x, ids, weights, bank)
        actual = candidate.forward(x, ids, weights, bank)
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_decode_accepts_strided_ids_and_weight_banks():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("SM89 FP8 kernel required")
    generator = torch.Generator().manual_seed(2136)
    bank = StagedExperts(
        w13_weight=torch.randn(8, 256, 512, generator=generator).to(torch.float8_e4m3fn).cuda()[..., ::2],
        w13_scale_inv=torch.rand(8, 2, 4, generator=generator).cuda()[..., ::2],
        w2_weight=torch.randn(8, 256, 256, generator=generator).to(torch.float8_e4m3fn).cuda()[..., ::2],
        w2_scale_inv=torch.rand(8, 2, 2, generator=generator).cuda()[..., ::2],
    )
    ids = torch.tensor([[3, 0, 2, 0, 6, 0, 1, 0]], device="cuda", dtype=torch.int32)[:, ::2]
    x = torch.randn(1, 256, generator=generator).bfloat16().cuda()
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device="cuda", dtype=torch.bfloat16)
    expected = TritonExpertBackend(decode_specialization=False).forward(x, ids, weights, bank)
    actual = TritonExpertBackend(decode_specialization=True).forward(x, ids, weights, bank)
    assert torch.equal(actual, expected)

    for count in (1, 2, 4, 8):
        rows = torch.randn(count, 256, generator=generator).bfloat16().cuda()
        route = torch.tensor(([3, 1, 3, 2, 7, 0, 6, 1][:count]), device="cuda").reshape(count, 1)
        route_weight = torch.rand(count, 1, generator=generator).bfloat16().cuda()
        backend = TritonExpertBackend()
        expected = backend.forward_pairs(rows, route, route_weight, bank)
        actual = backend.forward_decode_pairs(rows, route, route_weight, bank)
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pairs", [9, 16, 32, 64])
def test_experimental_verification_pairs_preserve_official_partials(pairs):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89-specific experiment")
    from drama_low_vram.fp8_decode import quantized_pairs

    rng = torch.Generator().manual_seed(993)
    hidden, intermediate, experts = 256, 256, 8
    bank = StagedExperts(
        (torch.randn(experts, intermediate * 2, hidden, generator=rng) * 30).to(torch.float8_e4m3fn).cuda(),
        (torch.rand(experts, intermediate * 2 // 128, hidden // 128, generator=rng) * 0.002).cuda(),
        (torch.randn(experts, hidden, intermediate, generator=rng) * 30).to(torch.float8_e4m3fn).cuda(),
        (torch.rand(experts, hidden // 128, intermediate // 128, generator=rng) * 0.002).cuda())
    x = torch.randn(pairs, hidden, generator=rng).bfloat16().cuda()
    ids = torch.randint(experts, (pairs, 1), generator=rng).cuda()
    weights = torch.rand(pairs, 1, generator=rng).bfloat16().cuda()
    backend = TritonExpertBackend()
    expected = backend.forward_pairs(x, ids, weights, bank)
    q, scales = backend.kernel.quantize_groups(backend.kernel.quant_kernel, x, hidden)
    actual = quantized_pairs(backend.kernel, q, scales, ids, weights, bank)
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_decode_falls_back_for_prefill_rows():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("SM89 FP8 kernel required")
    from unittest.mock import patch

    candidate = TritonExpertBackend(decode_specialization=True)
    x = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(2, 1, device="cuda", dtype=torch.int64)
    weights = torch.ones(2, 1, device="cuda")
    bank = StagedExperts(
        torch.empty(1, 256, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(1, 2, 1, device="cuda"),
        torch.empty(1, 128, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(1, 1, 1, device="cuda"),
    )
    expected = torch.full_like(x, 2)
    with patch.object(candidate.kernel, "fp8_moe_forward", return_value=expected) as original:
        assert candidate.forward(x, ids, weights, bank) is expected
    original.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_other_gpu_architectures_retain_general_kernel():
    from unittest.mock import patch

    candidate = TritonExpertBackend(decode_specialization=True)
    x = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    candidate._checked_devices.add(x.device)
    candidate._direct_decode_devices.clear()
    ids = torch.zeros(1, 1, device="cuda", dtype=torch.int64)
    weights = torch.ones(1, 1, device="cuda")
    bank = StagedExperts(
        torch.empty(1, 256, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(1, 2, 1, device="cuda"),
        torch.empty(1, 128, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(1, 1, 1, device="cuda"))
    with patch.object(candidate.kernel, "fp8_moe_forward", return_value=x) as general:
        assert candidate.forward(x, ids, weights, bank) is x
    general.assert_called_once()
