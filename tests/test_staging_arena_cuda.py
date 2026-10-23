"""Packed expert arenas match the official FP8 kernel on CUDA."""

import pytest
import torch
from torch import nn

from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    ExpertStore,
    install_expert_offload,
)


@pytest.fixture(scope="module")
def official_fp8():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fish_speech.models.text2semantic.fish_omni import fp8_moe

    if not fp8_moe.fp8_gemm_supported(["cuda:0"]):
        pytest.skip("The official FP8 kernel requires Triton and CUDA SM89+")
    return fp8_moe.fp8_moe_forward


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = True
        generator = torch.Generator().manual_seed(20261015)
        for name, rows, columns in (("w13", 256, 128), ("w2", 128, 128)):
            self.register_buffer(
                f"{name}_weight",
                torch.randint(
                    -3, 4, (16, rows, columns), generator=generator
                ).to(FP8_DTYPE),
            )
            self.register_buffer(
                f"{name}_scale_inv",
                torch.rand(
                    (16, rows // 128, columns // 128), generator=generator
                ) * 0.02 + 0.01,
            )

    def forward(self, *_args):
        raise AssertionError("Expert manager must intercept forwarding")


@pytest.mark.parametrize("capacity", [8, 2], ids=["combined", "partitioned"])
@pytest.mark.parametrize("resident_experts", [0, 8], ids=["mapped", "mixed-host"])
def test_packed_cuda_routes_match_official_fp8(
    official_fp8, capacity, resident_experts
):
    experts = Experts()
    backbone = nn.Sequential(experts)
    manager = install_expert_offload(
        backbone,
        backend="triton",
        max_staged_experts=capacity,
        max_pair_rows=64,
        cache_experts=2,
        combined_decode_bank=True,
        packed_staging=True,
    )
    store = ExpertStore.from_module(experts)
    expert_bytes = store.nbytes // store.num_experts
    manager.configure_host_residency(expert_bytes * resident_experts)
    generator = torch.Generator().manual_seed(411)
    x = torch.randn(1, 128, generator=generator).to("cuda:0", torch.bfloat16)
    weights = torch.softmax(torch.randn(1, 8, generator=generator), -1).to("cuda:0")
    device_buffers = {
        name: getattr(store, name).to("cuda:0") for name in EXPERT_BUFFER_NAMES
    }
    routes = (
        [1, 3, 5, 7, 9, 11, 13, 15],
        [5, 2, 3, 4, 1, 6, 8, 10],
        [3, 2, 4, 6, 8, 10, 12, 14],
        [3, 2, 1, 2, 3, 6, 6, 8],
    )
    try:
        # Repeating the sequence also checks cache eviction and arena reuse.
        for _ in range(2):
            for route in routes:
                ids = torch.tensor([route], device="cuda:0")
                expected = official_fp8(x, ids, weights, **device_buffers)
                actual = experts(x, ids, weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert bool(torch.isfinite(actual).all())
        manager.synchronize()
        stats = manager.stats
        assert stats.forward_calls == 2 * len(routes)
        assert stats.hot_cache_hits > 0
        assert stats.hot_cache_loads > 2
        assert stats.packed_staging_calls == stats.staging_calls > 0
        assert stats.h2d_copy_calls == stats.staging_calls
        assert stats.h2d_bytes == stats.expert_loads * expert_bytes
        assert stats.peak_staged_experts <= capacity
        assert stats.device_buffer_bytes == capacity * expert_bytes
        assert bool(stats.combined_decode_calls) == (capacity == 8)
        assert manager.pool._host_arena.shape == (capacity, expert_bytes)
        assert manager.pool._device_arena.shape == (capacity, expert_bytes)
        report = manager.host_residency_report()
        assert report["resident_experts"] == resident_experts
        if resident_experts:
            assert report["observed_routes"] == 2 * len(routes) * 8
            assert report["resident_hits"] > 0
            assert report["mapped_fallback_reads"] > 0
        else:
            assert report["observed_routes"] == report["expert_reads"] == 0
    finally:
        manager.close()
