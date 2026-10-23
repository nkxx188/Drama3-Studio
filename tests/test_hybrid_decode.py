"""Shared multi-layer ownership, exact decode and resource recovery."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest
import torch

from drama_low_vram.native_backend import NativeQuantizationBoundary, default_library_path
from drama_low_vram.offload import ExpertOffloadManager, FP8_DTYPE


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not default_library_path().is_file()
    or torch.cuda.get_device_capability() != (8, 9),
    reason="Optional native library and SM89 GPU required",
)


class Experts(torch.nn.Module):
    fp8 = True

    def __init__(self, seed):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        for name, tensor in (
            ("w13_weight", torch.randint(-3, 4, (8, 256, 128), generator=rng).to(FP8_DTYPE)),
            ("w13_scale_inv", torch.rand(8, 2, 1, generator=rng) * 0.03 + 0.01),
            ("w2_weight", torch.randint(-3, 4, (8, 128, 128), generator=rng).to(FP8_DTYPE)),
            ("w2_scale_inv", torch.rand(8, 1, 1, generator=rng) * 0.03 + 0.01),
        ):
            self.register_buffer(name, tensor)

    def forward(self, *_args):
        raise AssertionError("Manager must own expert execution")


@pytest.fixture
def manager():
    manager = ExpertOffloadManager(
        cache_experts=2, max_staged_experts=8, combined_decode_bank=True)
    for index in range(3):
        manager.attach(str(index), Experts(80 + index))
    try:
        yield manager
    finally:
        manager.close()


def inputs():
    x = torch.randn(1, 128, generator=torch.Generator().manual_seed(219)).bfloat16().cuda()
    ids = torch.tensor([[3, 1, 3, 7, 0, 5, 2, 4]], device=x.device)
    weights = torch.tensor([[0.1, 0.2, 0.05, 0.1, 0.15, 0.05, 0.15, 0.2]],
                           device=x.device, dtype=torch.bfloat16)
    return x, ids, weights


@pytest.mark.parametrize("jobs", [1, 4, 8])
def test_all_layers_share_workers_pool_and_workspace(manager, jobs):
    x, ids, weights = inputs()
    expected = [module(x, ids, weights) for module in manager.modules.values()]
    torch.cuda.synchronize()
    manager.configure_hybrid_decode(jobs, threads=2)
    hybrid = manager.hybrid_decode
    executor, pool = hybrid.executor._handle, manager.pool
    # One-row prefill does not implicitly become CPU decode.
    list(manager.modules.values())[0](x, ids, weights)
    assert hybrid.stats.calls == 0
    for repeat in range(3):
        with hybrid.decode():
            for index, module in enumerate(manager.modules.values()):
                actual = module(x, ids, weights)
                assert torch.equal(actual, expected[index])
        assert hybrid.executor._handle == executor and manager.pool is pool
        assert len(manager._hot_caches) == 3
        assert hybrid.stats.workspace_allocations == 1
    assert hybrid.stats.calls == 9 and hybrid.stats.cpu_pairs == 9 * jobs
    assert hybrid.report()["device_workspace_bytes"] == 2 * 8 * 128 * 4
    assert hybrid.report()["pinned_workspace_bytes"] < 8192
    assert manager.stats.hot_cache_capacity_experts == 6
    # Nested scopes restore the parent selection and calling threads are isolated.
    before = hybrid.stats.calls
    with hybrid.decode(layers=[0]):
        with hybrid.decode(layers=[]):
            list(manager.modules.values())[0](x, ids, weights)
        for module in manager.modules.values():
            module(x, ids, weights)
    assert hybrid.stats.calls == before + 1
    with hybrid.suspend(), hybrid.decode():
        assert not hybrid.eligible(list(manager.modules.values())[0], x, ids)
    with hybrid.decode():
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert not worker.submit(
                hybrid.eligible, list(manager.modules.values())[0], x, ids).result()


def test_workspace_survives_replanning_and_failure_without_stale_cache(manager):
    x, ids, weights = inputs()
    module = list(manager.modules.values())[0]
    expected = module(x, ids, weights)
    manager.configure_hybrid_decode(4, threads=2)
    hybrid = manager.hybrid_decode
    handle = hybrid.executor._handle
    with hybrid.decode():
        assert torch.equal(module(x, ids, weights), expected)
        manager.clear_hot_cache()
        with patch.object(manager.pool, "stage", side_effect=RuntimeError("injected DMA dispatch failure")):
            with pytest.raises(RuntimeError, match="injected DMA"):
                module(x, ids, weights)
        assert torch.equal(module(x, ids, weights), expected)
        invalid = ids.clone()
        invalid[0, 0] = 99
        with pytest.raises(ValueError, match="outside bank"):
            module(x, invalid, weights)
        with patch.object(hybrid.executor, "forward_quantized_pairs",
                          side_effect=RuntimeError("injected worker failure")):
            with pytest.raises(RuntimeError, match="injected worker"):
                module(x, ids, weights)
        assert torch.equal(module(x, ids, weights), expected)
    old_cache = manager._hot_caches[id(module)]
    manager.release_runtime_buffers()
    assert not hybrid._buffers and not old_cache._device and not manager._hot_caches
    manager.configure_staging(4, 64)
    manager.prepare_buffers(x.device)
    with torch.cuda.stream(torch.cuda.Stream()), hybrid.decode():
        assert torch.equal(module(x, ids, weights), expected)
    assert manager._hot_caches[id(module)] is not old_cache
    assert hybrid.executor._handle == handle
    assert hybrid.stats.workspace_allocations == 2
    assert manager.stats.device_buffer_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in manager.pool._device.values())
    manager.configure_hybrid_decode(0)
    assert manager.hybrid_decode is None and hybrid._closed and hybrid.executor._handle is None
    assert torch.equal(module(x, ids, weights), expected)


def test_insufficient_staging_falls_back_and_keeps_all_routes(manager):
    x, ids, weights = inputs()
    module = list(manager.modules.values())[0]
    expected = module(x, ids, weights)
    manager.configure_hybrid_decode(2, threads=2)
    manager.configure_staging(2, 64)
    hybrid = manager.hybrid_decode
    with hybrid.decode():
        assert torch.equal(module(x, ids, weights), expected)
    assert hybrid.stats.calls == 0 and hybrid.stats.fallbacks == 1


@pytest.mark.parametrize("synchronize_return", [False, True])
def test_completion_dependency_protects_scratch_across_streams_and_release(
    manager, synchronize_return,
):
    """Do not synchronize observations between calls: scratch stays in flight."""
    x, ids, weights = inputs()
    modules = list(manager.modules.values())
    expected = [module(x, ids, weights) for module in modules]
    torch.cuda.synchronize()
    manager.configure_hybrid_decode(2, threads=2)
    hybrid = manager.hybrid_decode
    hybrid.synchronize_return = synchronize_return
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    actual = []
    for repeat in range(4):
        for index, module in enumerate(modules):
            with torch.cuda.stream(streams[(repeat + index) % 2]), hybrid.decode():
                # Queue enough work to exercise reuse with a pending merge.
                torch.cuda._sleep(100000)
                actual.append((index, module(x, ids, weights)))
    # Release must drain pending GPU readers even with an asynchronous return.
    hybrid.release_runtime_buffers()
    assert not hybrid._pending and not hybrid._buffers
    torch.cuda.synchronize()
    for index, output in actual:
        assert torch.equal(output, expected[index])
    assert hybrid.stats.completion_host_waits == (12 if synchronize_return else 0)
    assert hybrid.stats.completion_stream_waits == (0 if synchronize_return else 11)
    with hybrid.decode():
        assert torch.equal(modules[0](x, ids, weights), expected[0])


@pytest.mark.parametrize("cpu_jobs,staging", [(4, 8), (4, 4), (8, 4)])
def test_guard_recomputes_cpu_slots_on_gpu_with_bounded_staging(manager, cpu_jobs, staging):
    x, ids, weights = inputs()
    module = list(manager.modules.values())[0]
    expected = module(x, ids, weights)
    manager.configure_hybrid_decode(cpu_jobs, threads=2)
    manager.configure_staging(staging, 64)
    hybrid = manager.hybrid_decode
    with hybrid.decode():
        with patch.object(
            hybrid.executor, "forward_quantized_pairs",
            side_effect=NativeQuantizationBoundary("GPU FP8 rounding boundary requires fallback"),
        ):
            actual = module(x, ids, weights)
        assert torch.equal(actual, expected)
        assert hybrid.stats.boundary_fallback_calls == 1
        assert hybrid.stats.boundary_recomputed_pairs == cpu_jobs
        assert hybrid.stats.gpu_pairs == 8 and hybrid.stats.partial_h2d_bytes == 0
        assert manager.stats.peak_staged_experts <= 8
        assert torch.equal(module(x, ids, weights), expected)


def test_close_waits_for_inflight_cpu_work_before_releasing_workspace(manager):
    x, ids, weights = inputs()
    module = list(manager.modules.values())[0]
    expected = module(x, ids, weights)
    manager.configure_hybrid_decode(4, threads=2)
    hybrid = manager.hybrid_decode
    entered, release = Event(), Event()
    original = hybrid._cpu

    def delayed_cpu(*args):
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Test worker was not released")
        original(*args)

    def forward():
        with hybrid.decode():
            return module(x, ids, weights)

    with patch.object(hybrid, "_cpu", delayed_cpu), ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(forward)
        try:
            assert entered.wait(timeout=10)
            closing = workers.submit(hybrid.close)
            assert not closing.done()
        finally:
            release.set()
        assert torch.equal(pending.result(timeout=10), expected)
        closing.result(timeout=10)
    assert hybrid._closed and not hybrid._buffers and hybrid.executor._handle is None
    manager.configure_hybrid_decode(0)
