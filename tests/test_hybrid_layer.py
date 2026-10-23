"""Route ownership and opt-in single-layer experiment contracts."""

import pytest
import torch

from drama_low_vram.hybrid_layer import HybridLayerExperiment, partition_routes
from drama_low_vram.native_backend import default_library_path
from drama_low_vram.offload import ExpertStore, FP8_DTYPE, TritonExpertBackend


@pytest.mark.parametrize("count", [0, 1, 2, 4, 8])
def test_partition_preserves_every_route_slot_once(count):
    ids = [6, 3, 6, 2, 1, 3, 7, 0]
    cpu, gpu = partition_routes(ids, {3: 0, 0: 1}, count)
    assert len(cpu) == count
    assert cpu == sorted(cpu) and gpu == sorted(gpu)
    assert sorted(cpu + gpu) == list(range(8))
    assert not set(cpu).intersection(gpu)
    if count <= 6:
        assert all(ids[slot] not in (3, 0) for slot in cpu)


def test_partition_prefers_cold_routes_before_hot_and_checks_bounds():
    assert partition_routes([1, 2, 3, 4], {4: 0}, 2) == ([1, 2], [0, 3])
    assert partition_routes([1, 2, 3, 4], {4: 0}, 4) == ([0, 1, 2, 3], [])
    with pytest.raises(ValueError, match="route count"):
        partition_routes([1], {}, 2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not default_library_path().is_file(),
    reason="Optional native DLL and CUDA required",
)
@pytest.mark.parametrize("cpu_launch", ["gather", "dma", "compute"])
def test_hybrid_serial_parallel_order_and_lifecycle(cpu_launch):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Official FP8 GPU kernel requires SM89")
    generator = torch.Generator().manual_seed(9281)
    store = ExpertStore(
        w13_weight=torch.randint(-3, 4, (8, 256, 128), generator=generator).to(FP8_DTYPE),
        w13_scale_inv=torch.full((8, 2, 1), 0.03125),
        w2_weight=torch.randint(-3, 4, (8, 128, 128), generator=generator).to(FP8_DTYPE),
        w2_scale_inv=torch.full((8, 1, 1), 0.03125),
    )
    x = torch.randn(1, 128, generator=generator).bfloat16().cuda()
    ids = torch.tensor([[3, 1, 7, 2, 6, 0, 4, 5]], device=x.device)
    weights = torch.full((1, 8), 0.125, device=x.device, dtype=torch.bfloat16)
    with HybridLayerExperiment(store, cpu_jobs=0) as baseline:
        expected, _ = baseline.forward(x, ids, weights)
        with baseline.pool.session(store, x.device):
            staged = baseline.pool.stage(store, ids.cpu().reshape(-1))
            official = TritonExpertBackend().forward(
                x, torch.arange(8, device=x.device).reshape(1, 8), weights, staged)
        torch.cuda.synchronize()
        assert torch.equal(expected, official)
    outputs = []
    for parallel in (False, True):
        with HybridLayerExperiment(
            store, cpu_jobs=2, threads=2, cache_experts=2, parallel=parallel,
            cpu_launch=cpu_launch,
        ) as layer:
            sequence = []
            for _ in range(3):
                output, report = layer.forward(x, ids, weights)
                sequence.append(output.clone())
                assert report["cpu_result_h2d_bytes"] == 2 * 128 * 4
                assert report["total_ms"] > 0
                assert sorted(report["cpu_slots"] + report["gpu_slots"]) == list(range(8))
            outputs.append(sequence)
        layer.close()
        with pytest.raises(RuntimeError, match="closed"):
            layer.forward(x, ids, weights)
    for serial, parallel in zip(*outputs, strict=True):
        assert torch.equal(serial, parallel)
