"""Empirical SM89 arithmetic is opt-in and checked against live official GPU ops."""

import pytest
import torch

from drama_low_vram.fp8_decode import build_gpu_silu_table
from drama_low_vram.native_backend import NativeExpertExecutor, NativeLibrary, default_library_path
from drama_low_vram.offload import ExpertStore, StagedExperts, TritonExpertBackend


@pytest.mark.skipif(
    not torch.cuda.is_available() or not default_library_path().is_file(),
    reason="SM89 CUDA and optional native DLL required",
)
@pytest.mark.parametrize("implementation,mask", [("scalar", 1), ("avx2", 2), ("avx512", 4)])
def test_sm89_partials_and_simd_match_official(implementation, mask):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Hardware-specific SM89 arithmetic")
    library = NativeLibrary()
    if not library.features & mask:
        pytest.skip("CPU ISA unavailable")
    generator = torch.Generator().manual_seed(28460)
    experts, hidden, intermediate = 4, 256, 256
    store = ExpertStore(
        w13_weight=(torch.randn(experts, 2 * intermediate, hidden, generator=generator) * 50)
        .clamp(-448, 448).to(torch.float8_e4m3fn),
        w13_scale_inv=torch.rand(experts, 2 * intermediate // 128, hidden // 128, generator=generator) * 0.001,
        w2_weight=(torch.randn(experts, hidden, intermediate, generator=generator) * 50)
        .clamp(-448, 448).to(torch.float8_e4m3fn),
        w2_scale_inv=torch.rand(experts, hidden // 128, intermediate // 128, generator=generator) * 0.001,
    )
    bank = StagedExperts(**{
        name: getattr(store, name).cuda()
        for name in ("w13_weight", "w13_scale_inv", "w2_weight", "w2_scale_inv")
    })
    table = build_gpu_silu_table()
    ids = torch.tensor([3, 1, 3, 0])
    weights = torch.tensor([0.4, 0.2, 0.3, 0.1]).bfloat16().float()
    gpu = TritonExpertBackend()
    with NativeExpertExecutor(
        store, implementation=implementation, arithmetic="sm89",
        silu_table=table, threads=3, library=library,
    ) as executor:
        for _ in range(3):
            x = torch.randn(1, hidden, generator=generator).bfloat16()
            xg = x.cuda()
            q, scales = gpu.kernel.quantize_groups(gpu.kernel.quant_kernel, xg, hidden)
            expected = gpu.forward_pairs(
                xg.repeat(4, 1), ids.cuda().reshape(4, 1),
                weights.cuda().reshape(4, 1), bank).cpu()
            trace = executor.trace_quantized_pairs(q.cpu(), scales.cpu(), ids, weights)
            actual = executor.forward_quantized_pairs(q.cpu(), scales.cpu(), ids, weights)
            assert torch.equal(trace["partials"], actual)
            assert torch.equal(actual, expected)
        # Mutating the caller's table must not alter the executor's borrowed copy.
        expected = actual.clone()
        table.zero_()
        assert torch.equal(
            executor.forward_quantized_pairs(q.cpu(), scales.cpu(), ids, weights), expected)
        with pytest.raises(ValueError, match="overlap"):
            executor.forward_quantized_pairs(
                q.cpu(), scales.cpu(), ids, weights,
                out=executor._silu_table[:ids.numel() * hidden].reshape(ids.numel(), hidden))


@pytest.mark.skipif(not default_library_path().is_file(), reason="Optional native DLL required")
def test_native_arithmetic_and_table_contract_validation():
    library = NativeLibrary()
    store = ExpertStore(
        torch.zeros(1, 256, 128, dtype=torch.float8_e4m3fn),
        torch.ones(1, 2, 1), torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
        torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="arithmetic"):
        NativeExpertExecutor(store, arithmetic="unknown", library=library)
    with pytest.raises(ValueError, match="SiLU table"):
        NativeExpertExecutor(store, silu_table=torch.zeros(128), library=library)
