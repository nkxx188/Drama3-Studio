"""Native byte contract, ordered W8A8 partials and executor lifecycle."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from drama_low_vram.fp8_cpu_reference import expert_pairs, quantize_fp8_blocks
from drama_low_vram.native_backend import (
    NativeExpertExecutor, NativeLibrary, NativeQuantizationBoundary, default_library_path,
)
from drama_low_vram.offload import ExpertStore, FP8_DTYPE


@pytest.fixture(scope="module")
def library():
    if not default_library_path().is_file():
        pytest.skip("Build the optional native library with tools/build_native.py")
    return NativeLibrary()


@pytest.fixture
def store():
    generator = torch.Generator().manual_seed(194)
    return ExpertStore(
        w13_weight=torch.randint(-3, 4, (4, 256, 128), generator=generator).to(FP8_DTYPE),
        w13_scale_inv=torch.full((4, 2, 1), 0.03125),
        w2_weight=torch.randint(-3, 4, (4, 128, 128), generator=generator).to(FP8_DTYPE),
        w2_scale_inv=torch.full((4, 1, 1), 0.03125),
    )


def test_all_fp8_bytes_decode_with_signed_zero_and_nan(library):
    raw = torch.arange(256, dtype=torch.uint8)
    actual = library.decode(raw)
    expected = raw.view(FP8_DTYPE).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert bool(torch.signbit(actual[128]))


@pytest.mark.parametrize("kind", ["random", "zero", "tiny", "saturated"])
def test_activation_quantization_exact_bytes_and_scales(library, kind):
    x = torch.randn(16, 128, generator=torch.Generator().manual_seed(22))
    if kind == "zero":
        x.zero_()
        x[0, :4] = torch.tensor([0., -0., 0., -0.])
    elif kind == "tiny":
        x *= 1e-20
    elif kind == "saturated":
        x *= 1e20
    x = x.to(torch.bfloat16)
    actual, scales = library.quantize(x)
    expected, expected_scales = quantize_fp8_blocks(x)
    assert torch.equal(actual, expected.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("implementation,mask", [("scalar", 1), ("avx2", 2), ("avx512", 4)])
@pytest.mark.parametrize("threads", [1, 3])
def test_ordered_duplicate_partials_match_explicit_contract(
    library, store, implementation, mask, threads
):
    if not library.features & mask:
        pytest.skip("CPU ISA unavailable")
    x = torch.linspace(-1, 1, 128).to(torch.bfloat16)
    ids = torch.tensor([3, 1, 3, 0])
    weights = torch.tensor([0.4, 0.2, 0.3, 0.1])
    expected = expert_pairs(store, x, ids, weights)
    with NativeExpertExecutor(
        store, threads=threads, implementation=implementation, library=library
    ) as executor:
        output = torch.empty(4, 128)
        for _ in range(3):
            actual = executor.forward_pairs(x, ids, weights, out=output)
            assert actual is output
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(actual[0] / 0.4, actual[2] / 0.3, rtol=1e-6, atol=1e-6)


def test_serialized_calls_and_error_recovery(library, store):
    x = torch.ones(128, dtype=torch.bfloat16)
    ids, weights = torch.tensor([2, 1]), torch.tensor([0.7, 0.3])
    with NativeExpertExecutor(store, threads=3, library=library) as executor:
        expected = executor.forward_pairs(x, ids, weights)
        with ThreadPoolExecutor(max_workers=3) as workers:
            results = list(workers.map(
                lambda _: executor.forward_pairs(x, ids, weights), range(9)
            ))
        for result in results:
            assert torch.equal(result, expected)
        with pytest.raises(RuntimeError, match="Expert ID"):
            executor.forward_pairs(x, torch.tensor([9]), torch.ones(1))
        invalid = x.clone()
        invalid[0] = float("nan")
        with pytest.raises(RuntimeError, match="Non-finite"):
            executor.forward_pairs(invalid, ids, weights)
        assert torch.equal(executor.forward_pairs(x, ids, weights), expected)
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.forward_pairs(x, ids, weights)


def test_invalid_types_and_missing_optional_library(library, store, tmp_path):
    with pytest.raises(FileNotFoundError, match="Optional native"):
        NativeLibrary(tmp_path / "missing.dll")
    with pytest.raises(ValueError, match="capacity"):
        NativeExpertExecutor(store, threads=True, library=library)
    with NativeExpertExecutor(store, library=library) as executor:
        with pytest.raises(ValueError, match="BF16"):
            executor.forward_pairs(torch.ones(128), torch.tensor([0]), torch.ones(1))
        with pytest.raises(ValueError, match="int64"):
            executor.forward_pairs(torch.ones(128, dtype=torch.bfloat16),
                                   torch.tensor([0], dtype=torch.int32), torch.ones(1))


def test_quantized_handoff_matches_raw_cpu_contract(library, store):
    x = torch.randn(128, generator=torch.Generator().manual_seed(44)).bfloat16()
    ids, weights = torch.tensor([3, 1, 0]), torch.tensor([0.5, 0.3, 0.2])
    q, scales = library.quantize(x)
    with NativeExpertExecutor(store, threads=3, library=library) as executor:
        expected = executor.forward_pairs(x, ids, weights)
        assert torch.equal(executor.forward_quantized_pairs(q, scales, ids, weights), expected)
        invalid = scales.clone().zero_()
        with pytest.raises(RuntimeError, match="scale"):
            executor.forward_quantized_pairs(q, invalid, ids, weights)
        assert torch.equal(executor.forward_quantized_pairs(q, scales, ids, weights), expected)


def test_trace_and_injected_intermediates_preserve_executor_arithmetic(library, store):
    x = torch.randn(128, generator=torch.Generator().manual_seed(219)).bfloat16()
    ids, weights = torch.tensor([2, 0, 2]), torch.tensor([0.5, 0.2, 0.3])
    q, scales = library.quantize(x)
    with NativeExpertExecutor(store, threads=3, library=library) as executor:
        expected = executor.forward_quantized_pairs(q, scales, ids, weights)
        trace = executor.trace_quantized_pairs(q, scales, ids, weights)
        assert torch.equal(trace["partials"], expected)
        assert torch.equal(trace["down"] * weights[:, None], expected)
        assert torch.equal(trace["gate_up"].bfloat16(), trace["gate_up_bf16"])
        shared_gu = executor.trace_quantized_pairs(
            q, scales, ids, weights, shared_gate_up=trace["gate_up_bf16"])
        shared_hq = executor.trace_quantized_pairs(
            q, scales, ids, weights, shared_hidden_q=trace["hidden_q"],
            shared_hidden_scales=trace["hidden_scales"])
        assert torch.equal(shared_gu["partials"], expected)
        assert torch.equal(shared_hq["partials"], expected)
        with pytest.raises(ValueError, match="together"):
            executor.trace_quantized_pairs(
                q, scales, ids, weights, shared_hidden_q=trace["hidden_q"])
        invalid_scales = trace["hidden_scales"].clone().zero_()
        with pytest.raises(RuntimeError, match="hidden FP8 scale"):
            executor.trace_quantized_pairs(
                q, scales, ids, weights, shared_hidden_q=trace["hidden_q"],
                shared_hidden_scales=invalid_scales)
        assert torch.equal(executor.forward_quantized_pairs(q, scales, ids, weights), expected)


@pytest.mark.parametrize("arithmetic", ["reference", "sm89"])
def test_native_worker_error_is_consumed_before_reuse(library, store, arithmetic):
    x = torch.ones(128, dtype=torch.bfloat16)
    ids, weights = torch.tensor([0, 1, 2]), torch.ones(3)
    original = store.w13_weight[1, 0, 0].clone()
    with NativeExpertExecutor(
        store, threads=3, arithmetic=arithmetic, library=library,
    ) as executor:
        expected = executor.forward_pairs(x, ids, weights)
        store.w13_weight[1, 0, 0] = float("nan")
        try:
            with pytest.raises(RuntimeError, match="Non-finite"):
                executor.forward_pairs(x, ids, weights)
        finally:
            store.w13_weight[1, 0, 0] = original
        assert torch.equal(executor.forward_pairs(x, ids, weights), expected)


def test_output_alias_is_rejected(library, store):
    storage = torch.ones(512, dtype=torch.float32)
    x = storage.view(torch.bfloat16)[:128]
    with NativeExpertExecutor(store, library=library) as executor:
        with pytest.raises(ValueError, match="overlap"):
            executor.forward_pairs(x, torch.tensor([0]), torch.ones(1),
                                   out=storage[:128].reshape(1, 128))


def test_one_executor_can_alternate_banks_without_rebinding_state(library, store):
    other = ExpertStore(
        store.w13_weight[:2].clone(), store.w13_scale_inv[:2].clone() * 2,
        store.w2_weight[:2].clone(), store.w2_scale_inv[:2].clone() * 0.5,
    )
    x = torch.randn(128, generator=torch.Generator().manual_seed(44)).bfloat16()
    ids, weights = torch.tensor([1, 0, 1]), torch.tensor([0.5, 0.3, 0.2])
    q, scales = library.quantize(x)
    with NativeExpertExecutor(store, threads=3, library=library) as executor:
        handle = executor._handle
        expected = {
            id(bank): executor.forward_quantized_pairs(q, scales, ids, weights, store=bank)
            for bank in (store, other)
        }
        assert not torch.equal(expected[id(store)], expected[id(other)])
        with ThreadPoolExecutor(max_workers=3) as workers:
            banks = [store, other] * 4
            results = list(workers.map(
                lambda bank: executor.forward_quantized_pairs(
                    q, scales, ids, weights, store=bank), banks))
        for bank, output in zip(banks, results, strict=True):
            assert torch.equal(output, expected[id(bank)])
            assert torch.equal(executor.forward_pairs(x, ids, weights, store=bank), output)
        assert executor.store is store and executor._handle == handle
        bad = ExpertStore(
            torch.zeros(1, 256, 256).to(FP8_DTYPE), torch.ones(1, 2, 2),
            torch.zeros(1, 256, 128).to(FP8_DTYPE), torch.ones(1, 2, 1))
        with pytest.raises(ValueError, match="geometry"):
            executor.forward_quantized_pairs(q, scales, ids, weights, store=bad)
        assert torch.equal(executor.forward_quantized_pairs(q, scales, ids, weights), expected[id(store)])


@pytest.mark.parametrize("midpoint", [42.0, 0.0009765625, -42.0])
def test_strict_hidden_quantization_guard_and_worker_recovery(library, store, midpoint):
    q, scales = library.quantize(torch.ones(128, dtype=torch.bfloat16))
    ids, weights = torch.tensor([1, 0]), torch.ones(2)
    gu = torch.zeros(2, 256, dtype=torch.bfloat16)
    gu[:, 128] = 448
    gu[:, 129] = midpoint
    with NativeExpertExecutor(
        store, threads=3, arithmetic="sm89", silu_table=torch.ones(65536),
        guard_fp8_boundaries=True, library=library,
    ) as executor:
        with pytest.raises(NativeQuantizationBoundary, match="rounding boundary"):
            executor.trace_quantized_pairs(q, scales, ids, weights, shared_gate_up=gu)
        gu[:, 129] = 40 if abs(midpoint) > 1 else 0.001953125
        trace = executor.trace_quantized_pairs(q, scales, ids, weights, shared_gate_up=gu)
        assert bool(torch.isfinite(trace["partials"]).all())


@pytest.mark.parametrize("implementation,mask", [("scalar", 1), ("avx2", 2), ("avx512", 4)])
def test_multiple_weight_scale_blocks_with_persistent_workers(library, implementation, mask):
    if not library.features & mask:
        pytest.skip("CPU ISA unavailable")
    generator = torch.Generator().manual_seed(882)
    store = ExpertStore(
        w13_weight=torch.randint(-3, 4, (3, 512, 256), generator=generator).to(FP8_DTYPE),
        w13_scale_inv=torch.rand(3, 4, 2, generator=generator) * 0.02 + 0.01,
        w2_weight=torch.randint(-3, 4, (3, 256, 256), generator=generator).to(FP8_DTYPE),
        w2_scale_inv=torch.rand(3, 2, 2, generator=generator) * 0.02 + 0.01,
    )
    x = torch.randn(256, generator=generator).bfloat16()
    ids, weights = torch.tensor([2, 0, 1, 2]), torch.tensor([0.4, 0.3, 0.2, 0.1])
    expected = expert_pairs(store, x, ids, weights)
    with NativeExpertExecutor(
        store, threads=3, implementation=implementation, library=library,
    ) as executor:
        for _ in range(8):
            torch.testing.assert_close(
                executor.forward_pairs(x, ids, weights), expected, rtol=2e-5, atol=2e-6
            )
