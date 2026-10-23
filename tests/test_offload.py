"""Small correctness checks; optional CUDA tests reuse the official FP8 kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    ExpertStore,
    FP8_DTYPE,
    TorchExpertBackend,
    dequantize_fp8_blocks,
    install_expert_offload,
    materialize_offloaded_backbone,
    select_fp8,
)


class FakeExperts(nn.Module):
    def __init__(self, experts=12, device="cpu", hidden=128, intermediate=128):
        super().__init__()
        self.fp8 = True
        self.num_experts = experts
        self.hidden = hidden
        self.intermediate = intermediate
        generator = torch.Generator().manual_seed(410)
        for name, rows, columns in (
            ("w13", 2 * intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
            if device == "meta":
                weight = torch.empty(
                    experts, rows, columns, dtype=FP8_DTYPE, device=device
                )
                scales = torch.empty(
                    experts, rows // 128, columns // 128, device=device
                )
            else:
                weight = torch.randint(
                    -3, 4, (experts, rows, columns), generator=generator
                ).to(FP8_DTYPE)
                scales = (
                    torch.rand(experts, rows // 128, columns // 128, generator=generator)
                    * 0.02
                    + 0.01
                )
            self.register_buffer(f"{name}_weight", weight)
            self.register_buffer(f"{name}_scale_inv", scales)

    def forward(self, x, ids, weights):
        raise AssertionError("Original expert forward must be replaced")


def dense_reference(experts, x, ids, weights):
    out = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            w13 = dequantize_fp8_blocks(
                select_fp8(experts.w13_weight, expert),
                experts.w13_scale_inv[expert],
                x.dtype,
            )
            w2 = dequantize_fp8_blocks(
                select_fp8(experts.w2_weight, expert),
                experts.w2_scale_inv[expert],
                x.dtype,
            )
            gate, up = F.linear(x[row], w13).chunk(2, dim=-1)
            out[row] += F.linear(F.silu(gate) * up, w2).float() * weights[row, slot]
    return out.to(x.dtype)


class RecordingBackend(TorchExpertBackend):
    def __init__(self):
        self.staged_counts = []
        self.pair_rows = []
        self.routed_pairs = 0

    def forward(self, x, ids, weights, staged):
        self.staged_counts.append(staged.count)
        return super().forward(x, ids, weights, staged)

    def forward_pairs(self, x, ids, weights, staged):
        self.staged_counts.append(staged.count)
        self.pair_rows.append(x.shape[0])
        self.routed_pairs += ids.numel()
        return super().forward_pairs(x, ids, weights, staged)

class OrderedContributionBackend(TorchExpertBackend):
    """Distinct magnitudes make a change in route reduction order observable."""

    def forward_pairs(self, x, ids, weights, staged):
        fingerprints = staged.w13_scale_inv[ids.reshape(-1), 0, 0].long()
        contributions = torch.tensor(
            [1.0, 3.0, 1e20, -1e20], device=x.device
        )
        return (
            contributions[fingerprints - 1, None]
            .expand(-1, x.shape[-1])
            .float()
            * weights.float()
        )


class OffloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(530)
        self.experts = FakeExperts()
        self.backbone = nn.Module()
        self.backbone.add_module("experts", self.experts)

    def attach(self, **kwargs):
        manager = install_expert_offload(
            self.backbone, pin_memory=False, **kwargs
        )
        self.addCleanup(manager.close)
        return manager

    def test_decode_keeps_all_eight_routes_and_state_keys(self):
        ids = torch.tensor([[11, 7, 3, 9, 0, 6, 2, 4]])
        weights = torch.tensor([[0.4, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04]])
        x = torch.randn(1, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        keys = list(self.backbone.state_dict())
        original_pointers = {
            name: getattr(self.experts, name).data_ptr() for name in EXPERT_BUFFER_NAMES
        }
        backend = RecordingBackend()
        manager = self.attach(backend=backend, max_staged_experts=8)
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(backend.routed_pairs, 8)
        self.assertEqual(manager.stats.expert_loads, 8)
        self.assertEqual(manager.stats.peak_staged_experts, 8)
        self.assertEqual(list(self.backbone.state_dict()), keys)
        for name, pointer in original_pointers.items():
            self.assertEqual(getattr(self.experts, name).data_ptr(), pointer)
            self.assertEqual(getattr(self.experts, name).device.type, "cpu")

    def test_prefill_partitions_union_without_dropping_routes(self):
        ids = torch.tensor(
            [
                [11, 1, 4, 7, 2, 8, 0, 10],
                [2, 9, 6, 1, 3, 11, 5, 7],
                [8, 4, 0, 10, 6, 9, 3, 5],
                [7, 2, 1, 3, 4, 5, 10, 11],
            ]
        )
        weights = torch.softmax(torch.randn_like(ids, dtype=torch.float32), dim=-1)
        x = torch.randn(4, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        backend = RecordingBackend()
        manager = self.attach(
            backend=backend, max_staged_experts=3, max_pair_rows=5
        )
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(backend.routed_pairs, ids.numel())
        self.assertLessEqual(max(backend.staged_counts), 3)
        self.assertLessEqual(max(backend.pair_rows), 5)
        self.assertEqual(manager.stats.expert_loads, 12)
        self.assertEqual(manager.stats.staging_calls, 4)
        bytes_per_expert = manager.stats.cpu_expert_bytes // 12
        self.assertEqual(manager.stats.device_buffer_bytes, bytes_per_expert * 3)

    def test_resident_union_splits_activation_rows_without_reloading(self):
        ids = torch.tensor([[8, 4]] * 7)
        weights = torch.tensor([[0.3, 0.7]] * 7)
        x = torch.randn(7, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        backend = RecordingBackend()
        manager = self.attach(
            backend=backend, max_staged_experts=2, max_pair_rows=4
        )
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(manager.stats.staging_calls, 1)
        self.assertEqual(backend.routed_pairs, 14)
        self.assertLessEqual(max(backend.pair_rows), 4)

    def test_reuses_identical_staging_group(self):
        ids = torch.tensor([[8, 4]] * 2)
        weights = torch.tensor([[0.3, 0.7]] * 2)
        x = torch.randn(2, 128)
        backend = RecordingBackend()
        manager = self.attach(
            backend=backend, max_staged_experts=2, max_pair_rows=4
        )
        self.experts(x, ids, weights)
        self.experts(x, ids, weights)
        self.assertEqual(manager.stats.staging_calls, 1)
        self.assertEqual(manager.stats.expert_loads, 2)
        self.assertEqual(manager.stats.expert_cache_hits, 2)

    def test_does_not_reuse_identical_routes_from_another_layer(self):
        second = FakeExperts()
        self.backbone.add_module("second", second)
        manager = install_expert_offload(
            self.backbone, backend="torch", max_staged_experts=2, pin_memory=False
        )
        self.addCleanup(manager.close)
        ids = torch.tensor([[8, 4]])
        weights = torch.tensor([[0.3, 0.7]])
        x = torch.randn(1, 128)
        self.experts(x, ids, weights)
        second(x, ids, weights)
        self.assertEqual(manager.stats.staging_calls, 2)
        self.assertEqual(manager.stats.expert_cache_hits, 0)

    def test_per_layer_hot_cache_reuses_only_same_layer(self):
        second = FakeExperts()
        self.backbone.add_module("second", second)
        manager = install_expert_offload(
            self.backbone,
            backend="torch",
            max_staged_experts=2,
            cache_experts=2,
            pin_memory=False,
        )
        self.addCleanup(manager.close)
        ids = torch.tensor([[8, 4]])
        weights = torch.tensor([[0.3, 0.7]])
        x = torch.randn(1, 128)
        first_expected = dense_reference(self.experts, x, ids, weights)
        second_expected = dense_reference(second, x, ids, weights)
        torch.testing.assert_close(self.experts(x, ids, weights), first_expected)
        torch.testing.assert_close(self.experts(x, ids, weights), first_expected)
        torch.testing.assert_close(second(x, ids, weights), second_expected)
        self.assertEqual(manager.stats.hot_cache_loads, 4)
        self.assertEqual(manager.stats.hot_cache_hits, 2)
        self.assertEqual(manager.stats.staging_calls, 2)
        self.assertEqual(len(manager._hot_caches), 2)

    def test_hot_cache_alias_and_conflicting_capacity_are_explicit(self):
        manager = self.attach(
            backend="torch",
            max_staged_experts=2,
            hot_cache_experts=2,
        )
        self.assertEqual(manager.cache_experts, 2)
        conflict_backbone = nn.Module()
        conflict_backbone.add_module("experts", FakeExperts())
        with self.assertRaisesRegex(ValueError, "must match"):
            install_expert_offload(
                conflict_backbone,
                backend="torch",
                cache_experts=1,
                hot_cache_experts=2,
            )

    def test_hot_cache_bytes_are_bounded_per_layer_and_released(self):
        second = FakeExperts()
        self.backbone.add_module("second", second)
        manager = install_expert_offload(
            self.backbone,
            backend="torch",
            max_staged_experts=2,
            cache_experts=2,
            pin_memory=False,
        )
        ids = torch.tensor([[8, 4]])
        weights = torch.tensor([[0.3, 0.7]])
        x = torch.randn(1, 128)
        self.experts(x, ids, weights)
        second(x, ids, weights)
        bytes_per_expert = sum(
            getattr(self.experts, name)[0].numel()
            * getattr(self.experts, name).element_size()
            for name in EXPERT_BUFFER_NAMES
        )
        self.assertEqual(manager.stats.hot_cache_capacity_experts, 4)
        self.assertEqual(manager.stats.hot_cache_bytes, bytes_per_expert * 4)
        manager.close()
        self.assertEqual(manager.stats.hot_cache_bytes, 0)
        self.assertEqual(manager.stats.hot_cache_capacity_experts, 0)

    def test_cache_smaller_than_topk_handles_hot_and_cold_routes(self):
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        weights = torch.softmax(torch.randn(1, 6), dim=-1)
        x = torch.randn(1, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        backend = RecordingBackend()
        manager = self.attach(
            backend=backend,
            max_staged_experts=2,
            cache_experts=2,
            max_pair_rows=3,
        )
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(backend.routed_pairs, ids.numel())
        self.assertEqual(manager.stats.hot_cache_loads, 2)
        self.assertEqual(manager.stats.staging_calls, 3)
        self.assertLessEqual(manager.stats.peak_staged_experts, 2)
        first_loads = manager.stats.expert_loads
        torch.testing.assert_close(
            self.experts(x, ids, weights), expected, rtol=1e-5, atol=1e-6
        )
        self.assertEqual(manager.stats.hot_cache_hits, 2)
        self.assertEqual(manager.stats.hot_cache_loads, 2)
        self.assertEqual(manager.stats.expert_loads - first_loads, 4)
        self.assertEqual(backend.routed_pairs, ids.numel() * 2)

    def test_prefill_reads_partial_cache_without_replacing_decode_hotspots(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2, max_pair_rows=3
        )
        self.experts(
            torch.randn(1, 128), torch.tensor([[8, 4]]), torch.tensor([[0.6, 0.4]])
        )
        ids = torch.tensor([[0, 8, 1, 4], [8, 2, 4, 3]])
        weights = torch.softmax(torch.randn(2, 4), dim=-1)
        x = torch.randn(2, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        cache = manager._hot_caches[id(self.experts)]
        self.assertEqual(set(cache._loaded), {8, 4})
        self.assertEqual(manager.stats.hot_cache_loads, 2)
        self.assertEqual(manager.stats.hot_cache_hits, 2)

    def test_prefill_does_not_populate_an_empty_cache(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        ids = torch.tensor([[8, 4], [4, 8]])
        x = torch.randn(2, 128)
        weights = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(
            actual, dense_reference(self.experts, x, ids, weights)
        )
        self.assertEqual(manager.stats.hot_cache_loads, 0)
        self.assertFalse(manager._hot_caches[id(self.experts)]._loaded)

    def test_duplicate_routes_with_partial_cache_preserve_every_pair(self):
        ids = torch.tensor([[1, 7, 1, 2, 7, 2, 4, 4]])
        weights = torch.softmax(torch.randn(1, 8), dim=-1)
        x = torch.randn(1, 128)
        backend = RecordingBackend()
        manager = self.attach(
            backend=backend,
            max_staged_experts=1,
            cache_experts=1,
            max_pair_rows=2,
        )
        expected = dense_reference(self.experts, x, ids, weights)
        for _ in range(2):
            torch.testing.assert_close(
                self.experts(x, ids, weights), expected, rtol=1e-5, atol=1e-6
            )
        self.assertEqual(backend.routed_pairs, 16)
        self.assertEqual(manager.stats.hot_cache_loads, 1)
        self.assertEqual(manager.stats.hot_cache_hits, 1)
        self.assertLessEqual(max(backend.pair_rows), 2)

    def test_partial_cache_keeps_layers_separate_during_eviction(self):
        second = FakeExperts()
        second.w13_scale_inv.mul_(2)
        self.backbone.add_module("second", second)
        manager = install_expert_offload(
            self.backbone,
            backend="torch",
            max_staged_experts=1,
            cache_experts=1,
            pin_memory=False,
        )
        self.addCleanup(manager.close)
        routes = (torch.tensor([[8, 4]]), torch.tensor([[7, 11]]))
        weights = torch.tensor([[0.7, 0.3]])
        x = torch.randn(1, 128)
        for ids in (*routes, routes[0]):
            for module in (self.experts, second):
                torch.testing.assert_close(
                    module(x, ids, weights),
                    dense_reference(module, x, ids, weights),
                    rtol=1e-5,
                    atol=1e-6,
                )
        self.assertEqual(len(manager._hot_caches), 2)
        self.assertEqual(manager.stats.hot_cache_loads, 6)
        for cache in manager._hot_caches.values():
            self.assertEqual(set(cache._loaded), {8})

    def test_partitioned_fp32_reduction_retains_original_router_order(self):
        self.experts.w13_scale_inv[:, 0, 0] = torch.arange(1, 13)
        ids = torch.tensor([[2, 0, 3, 1]])
        weights = torch.ones(1, 4)
        x = torch.zeros(1, 128)
        expected = torch.tensor([1e20, 1.0, -1e20, 3.0]).sum().expand_as(x)
        manager = self.attach(
            backend=OrderedContributionBackend(), max_staged_experts=1, max_pair_rows=2
        )
        torch.testing.assert_close(
            self.experts(x, ids, weights), expected, rtol=0, atol=0
        )
        manager.configure_hot_cache(1)
        torch.testing.assert_close(
            self.experts(x, ids, weights), expected, rtol=0, atol=0
        )

    def test_prepare_and_resize_buffers_do_not_load_expert_weights(self):
        second = FakeExperts()
        self.backbone.add_module("second", second)
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        stats = manager.prepare_buffers("cpu")
        per_expert = stats["cpu_expert_bytes"] // 24
        self.assertEqual(stats["hot_cache_bytes"], per_expert * 4)
        self.assertEqual(stats["device_buffer_bytes"], per_expert * 2)
        self.assertEqual(stats["expert_loads"], 0)
        manager.configure_hot_cache(1)
        manager.configure_staging(1, 3)
        stats = manager.prepare_buffers("cpu")
        self.assertEqual(stats["hot_cache_bytes"], per_expert * 2)
        self.assertEqual(stats["device_buffer_bytes"], per_expert)
        self.assertEqual(stats["expert_loads"], 0)
        self.assertEqual(manager.max_pair_rows, 3)
        self.assertEqual(stats["cpu_expert_bytes"], per_expert * 24)

    def test_unchanged_configuration_keeps_warm_cache_and_buffer_addresses(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        x = torch.randn(1, 128)
        ids = torch.tensor([[7, 3]])
        weights = torch.tensor([[0.7, 0.3]])
        first = self.experts(x, ids, weights)
        cache = manager._hot_caches[id(self.experts)]
        pool = manager.pool
        cache_addresses = {name: t.data_ptr() for name, t in cache._device.items()}
        staging_addresses = {name: t.data_ptr() for name, t in pool._device.items()}
        loads = manager.stats.expert_loads
        with patch.object(manager, "synchronize", side_effect=AssertionError("no wait")):
            manager.configure_hot_cache(2)
            manager.configure_staging(2, manager.max_pair_rows)
        manager.prepare_buffers("cpu")
        second = self.experts(x, ids, weights)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(manager.stats.expert_loads, loads)
        self.assertEqual(manager.stats.hot_cache_hits, 2)
        self.assertIs(manager.pool, pool)
        self.assertIs(manager._hot_caches[id(self.experts)], cache)
        self.assertEqual(cache_addresses, {name: t.data_ptr() for name, t in cache._device.items()})
        self.assertEqual(staging_addresses, {name: t.data_ptr() for name, t in pool._device.items()})

    def test_pair_limit_change_reuses_staging_contents(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        x = torch.randn(1, 128)
        ids = torch.tensor([[7, 3]])
        weights = torch.tensor([[0.7, 0.3]])
        first = self.experts(x, ids, weights)
        pool = manager.pool
        manager.configure_staging(2, 1)
        second = self.experts(x, ids, weights)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertIs(manager.pool, pool)
        self.assertEqual(manager.max_pair_rows, 1)
        self.assertEqual(manager.stats.staging_calls, 1)
        self.assertEqual(manager.stats.expert_cache_hits, 2)

    def test_staging_resize_retains_warm_layer_cache(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        x = torch.randn(1, 128)
        ids = torch.tensor([[7, 3]])
        weights = torch.tensor([[0.7, 0.3]])
        expected = self.experts(x, ids, weights)
        cache = manager._hot_caches[id(self.experts)]
        pool = manager.pool
        manager.configure_staging(1, 1)
        self.assertIsNot(manager.pool, pool)
        self.assertIs(cache.pool, manager.pool)
        torch.testing.assert_close(self.experts(x, ids, weights), expected, rtol=0, atol=0)
        self.assertEqual(manager.stats.expert_loads, 2)
        self.assertEqual(manager.stats.hot_cache_hits, 2)

    def test_explicit_clear_drops_hot_experts_but_retains_capacity_and_staging(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        manager.prepare_buffers("cpu")
        pool = manager.pool
        staging = dict(pool._device)
        manager.clear_hot_cache()
        self.assertEqual(manager.cache_experts, 2)
        self.assertFalse(manager._hot_caches)
        self.assertEqual(manager.stats.hot_cache_bytes, 0)
        self.assertEqual(manager.stats.hot_cache_capacity_experts, 0)
        self.assertEqual(staging, pool._device)
        self.assertGreater(manager.prepare_buffers("cpu")["hot_cache_bytes"], 0)

    def test_explicit_release_reclaims_all_runtime_buffers_and_can_prepare_again(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2, max_pair_rows=3
        )
        manager.prepare_buffers("cpu")
        manager.release_runtime_buffers()
        self.assertEqual((manager.cache_experts, manager.max_staged_experts, manager.max_pair_rows), (2, 2, 3))
        self.assertFalse(manager._hot_caches)
        self.assertFalse(manager.pool._host)
        self.assertFalse(manager.pool._device)
        self.assertEqual(manager.stats.device_buffer_bytes, 0)
        self.assertEqual(manager.stats.hot_cache_bytes, 0)
        stats = manager.prepare_buffers("cpu")
        self.assertGreater(stats["device_buffer_bytes"], 0)
        self.assertGreater(stats["hot_cache_bytes"], 0)
        self.assertEqual(stats["expert_loads"], 0)

    def test_transfer_breakdown_does_not_time_cpu_copies_as_h2d(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        self.experts(torch.randn(1, 128), torch.tensor([[7, 3]]), torch.ones(1, 2))
        stats = manager.stats
        self.assertGreater(stats.host_gather_seconds, 0)
        self.assertGreaterEqual(stats.host_staging_seconds, stats.host_gather_seconds)
        self.assertGreater(stats.routing_host_seconds, 0)
        self.assertEqual(stats.dma_source_wait_seconds, 0)
        self.assertEqual(stats.h2d_submit_seconds, 0)
        self.assertEqual(stats.h2d_seconds, 0)
        self.assertEqual(stats.h2d_bytes, 0)

    def test_source_wait_and_device_copy_time_are_separate(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        completion = SimpleNamespace(synchronize=lambda: None)
        start = SimpleNamespace(elapsed_time=lambda end: 25.0)
        manager.pool._copy_done = completion
        manager.pool._copy_start = start
        with patch("drama_low_vram.offload.time.perf_counter", side_effect=[10.0, 10.002]):
            manager.pool._finish_copy()
        self.assertAlmostEqual(manager.stats.dma_source_wait_seconds, 0.002)
        self.assertAlmostEqual(manager.stats.h2d_seconds, 0.025)
        self.assertIsNone(manager.pool._copy_done)
        self.assertIsNone(manager.pool._copy_start)

    def test_host_residency_mixed_sources_keep_all_routes_and_observe_once(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=1
        )
        expert_bytes = ExpertStore.from_module(self.experts).nbytes // 12
        report = manager.configure_host_residency(expert_bytes * 3)
        self.assertEqual(report["resident_experts"], 3)
        ids = torch.tensor([[3, 1, 8, 1, 0, 6, 2, 4]])
        weights = torch.softmax(torch.randn(1, 8), -1)
        x = torch.randn(1, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        for _ in range(2):
            torch.testing.assert_close(
                self.experts(x, ids, weights), expected, rtol=1e-5, atol=1e-6
            )
        report = manager.host_residency_report()
        self.assertEqual(report["observed_routes"], 16)
        self.assertGreater(report["resident_hits"], 0)
        self.assertGreater(report["mapped_fallback_reads"], 0)
        self.assertEqual(
            report["expert_reads"], report["resident_hits"] + report["mapped_fallback_reads"]
        )

    def test_host_rank_observes_gpu_hits_and_rebalances_only_at_explicit_boundary(self):
        manager = self.attach(
            backend="torch", max_staged_experts=1, cache_experts=1
        )
        expert_bytes = ExpertStore.from_module(self.experts).nbytes // 12
        manager.configure_host_residency(expert_bytes)
        x = torch.randn(1, 128)
        ids, weights = torch.tensor([[7]]), torch.ones(1, 1)
        for _ in range(3):
            self.experts(x, ids, weights)
        resident = manager._host_residency
        self.assertEqual(resident.resident_keys, ((0, 0),))
        self.assertEqual(resident.stats.observed_routes, 3)
        self.assertEqual(resident.stats.expert_reads, 1)
        manager.configure_host_residency(expert_bytes)
        self.assertEqual(resident.resident_keys, ((0, 0),))
        manager.rebalance_host_residency()
        self.assertEqual(resident.resident_keys, ((0, 7),))
        self.assertEqual(resident.stats.observed_routes, 3)

    def test_host_residency_survives_gpu_release_and_staging_resize(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=1
        )
        expert_bytes = ExpertStore.from_module(self.experts).nbytes // 12
        manager.configure_host_residency(expert_bytes * 2)
        resident = manager._host_residency
        arena = resident._arena["w13_weight"]
        manager.configure_staging(1, 1)
        self.assertIs(manager.pool.host_residency, resident)
        manager.prepare_buffers("cpu")
        manager.release_runtime_buffers()
        self.assertIs(manager.pool.host_residency, resident)
        self.assertIs(resident._arena["w13_weight"], arena)
        x, ids, weights = torch.randn(1, 128), torch.tensor([[1, 8]]), torch.ones(1, 2)
        torch.testing.assert_close(
            self.experts(x, ids, weights),
            dense_reference(self.experts, x, ids, weights),
            rtol=1e-5, atol=1e-6,
        )
        self.assertEqual(resident.stats.resident_hits, 1)
        self.assertEqual(resident.stats.mapped_fallback_reads, 1)

    def test_zero_host_budget_uses_original_gather_without_route_observation(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        manager.configure_host_residency(0)
        resident = manager._host_residency
        with patch.object(resident, "copy_to", side_effect=AssertionError("disabled tier")):
            self.experts(torch.randn(1, 128), torch.tensor([[1, 8]]), torch.ones(1, 2))
        self.assertEqual(resident.stats.requests, 0)
        self.assertEqual(resident.stats.observed_routes, 0)

    def test_changed_host_source_requires_reconfiguration_before_using_cached_bytes(self):
        manager = self.attach(
            backend="torch", max_staged_experts=2, cache_experts=2
        )
        expert_bytes = ExpertStore.from_module(self.experts).nbytes // 12
        manager.configure_host_residency(expert_bytes * 2)
        x, ids, weights = torch.randn(1, 128), torch.tensor([[1, 0]]), torch.ones(1, 2)
        self.experts(x, ids, weights)
        self.experts.w2_scale_inv = self.experts.w2_scale_inv.clone().mul_(2)
        with self.assertRaisesRegex(RuntimeError, "reconfigure host residency"):
            self.experts(x, ids, weights)
        manager.configure_host_residency(expert_bytes * 2)
        torch.testing.assert_close(
            self.experts(x, ids, weights),
            dense_reference(self.experts, x, ids, weights),
            rtol=1e-5, atol=1e-6,
        )

    def test_manager_close_releases_host_residency_sources(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        expert_bytes = ExpertStore.from_module(self.experts).nbytes // 12
        manager.configure_host_residency(expert_bytes)
        resident = manager._host_residency
        manager.close()
        self.assertFalse(manager.host_residency_report()["configured"])
        self.assertFalse(resident._arena)
        self.assertFalse(resident._stores)
        self.assertIsNone(manager.pool.host_residency)

    def test_prepare_cache_allocation_failure_rolls_back_cache_bytes(self):
        second = FakeExperts()
        self.backbone.add_module("second", second)
        manager = self.attach(
            backend="torch", max_staged_experts=1, cache_experts=2
        )
        real_empty = torch.empty
        cache_allocations = 0

        def fail_second_cache(*args, **kwargs):
            nonlocal cache_allocations
            shape = args[0] if args else kwargs.get("size")
            if shape[0] == 2:
                cache_allocations += 1
                if cache_allocations == 6:
                    raise torch.OutOfMemoryError("synthetic cache allocation failure")
            return real_empty(*args, **kwargs)

        with patch(
            "drama_low_vram.offload.torch.empty", side_effect=fail_second_cache
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                manager.prepare_buffers("cpu")
        self.assertEqual(manager.stats.hot_cache_bytes, 0)
        self.assertEqual(manager.stats.hot_cache_capacity_experts, 0)
        self.assertFalse(manager._hot_caches)
        manager.configure_hot_cache(1)
        self.assertGreater(manager.prepare_buffers("cpu")["hot_cache_bytes"], 0)

    def test_pinned_host_allocation_failure_falls_back_to_pageable_staging(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        manager.pool.pin_memory = True
        real_empty = torch.empty

        def reject_pinning(*args, **kwargs):
            if kwargs.get("pin_memory"):
                raise RuntimeError("synthetic Windows pinned host-lock allocation limit")
            # The allocation policy can be verified without an actual CUDA GPU.
            if torch.device(kwargs.get("device", "cpu")).type == "cuda":
                kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        with patch("drama_low_vram.offload.torch.empty", side_effect=reject_pinning):
            manager.pool._allocate(ExpertStore.from_module(self.experts), torch.device("cuda:0"))
        self.assertFalse(manager.pool.pin_memory)
        self.assertEqual(manager.stats.pinned_allocation_fallbacks, 1)
        self.assertIn("host-lock allocation limit", manager.pool.pin_memory_error)
        self.assertEqual(manager.stats.pinned_bytes, 0)
        self.assertGreater(manager.stats.device_buffer_bytes, 0)
        for tensor in manager.pool._host.values():
            self.assertFalse(tensor.is_pinned())

    def test_device_oom_is_not_treated_as_pinned_host_failure(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        manager.pool.pin_memory = True
        real_empty = torch.empty

        def reject_device_allocation(*args, **kwargs):
            if torch.device(kwargs.get("device", "cpu")).type == "cuda":
                raise torch.OutOfMemoryError("synthetic GPU buffer allocation failure")
            kwargs["pin_memory"] = False
            return real_empty(*args, **kwargs)

        with patch(
            "drama_low_vram.offload.torch.empty",
            side_effect=reject_device_allocation,
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "GPU buffer"):
                manager.pool._allocate(
                    ExpertStore.from_module(self.experts), torch.device("cuda:0")
                )
        self.assertEqual(manager.stats.pinned_allocation_fallbacks, 0)
        self.assertTrue(manager.pool.pin_memory)
        self.assertIsNone(manager.pool._target)
        self.assertFalse(manager.pool._host)
        self.assertFalse(manager.pool._device)

    def test_pageable_host_failure_is_propagated_without_retry(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        with patch(
            "drama_low_vram.offload.torch.empty",
            side_effect=RuntimeError("synthetic physical RAM allocation failure"),
        ) as allocator:
            with self.assertRaisesRegex(RuntimeError, "physical RAM"):
                manager.pool._allocate(
                    ExpertStore.from_module(self.experts), torch.device("cpu")
                )
        self.assertEqual(allocator.call_count, 1)
        self.assertEqual(manager.stats.pinned_allocation_fallbacks, 0)

    def test_unrelated_pinned_allocator_error_is_not_hidden(self):
        manager = self.attach(backend="torch", max_staged_experts=2)
        manager.pool.pin_memory = True
        with patch(
            "drama_low_vram.offload.torch.empty",
            side_effect=RuntimeError("synthetic unsupported shape"),
        ) as allocator:
            with self.assertRaisesRegex(RuntimeError, "unsupported shape"):
                manager.pool._allocate(
                    ExpertStore.from_module(self.experts), torch.device("cuda:0")
                )
        self.assertEqual(allocator.call_count, 1)
        self.assertEqual(manager.stats.pinned_allocation_fallbacks, 0)

    def test_duplicate_routes_accumulate_instead_of_overwriting(self):
        ids = torch.tensor([[1, 1, 7, 7], [2, 1, 2, 7]])
        weights = torch.tensor([[0.4, 0.1, 0.2, 0.3], [0.2, 0.3, 0.1, 0.4]])
        x = torch.randn(2, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        backend = RecordingBackend()
        self.attach(backend=backend, max_staged_experts=1, max_pair_rows=2)
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(backend.routed_pairs, 8)

    def test_small_pair_limit_also_splits_one_token(self):
        ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        weights = torch.full((1, 8), 0.125)
        x = torch.randn(1, 128)
        expected = dense_reference(self.experts, x, ids, weights)
        backend = RecordingBackend()
        self.attach(backend=backend, max_staged_experts=8, max_pair_rows=3)
        actual = self.experts(x, ids, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(backend.routed_pairs, 8)
        self.assertLessEqual(max(backend.pair_rows), 3)

    def test_out_of_range_ids_fail_before_loading_any_expert(self):
        manager = self.attach(backend="torch")
        with self.assertRaisesRegex(ValueError, "Expert IDs must be in"):
            self.experts(torch.randn(1, 128), torch.tensor([[12]]), torch.ones(1, 1))
        self.assertEqual(manager.stats.staging_calls, 0)

    def test_close_restores_original_forward_and_releases_pool(self):
        original = self.experts.forward
        manager = self.attach(backend="torch")
        self.experts(torch.randn(1, 128), torch.tensor([[1]]), torch.ones(1, 1))
        self.assertGreater(manager.stats.device_buffer_bytes, 0)
        manager.close()
        self.assertEqual(self.experts.forward, original)
        self.assertEqual(manager.stats.device_buffer_bytes, 0)

    def test_meta_placement_preserves_fp8_and_rejects_live_model(self):
        backbone = nn.Module()
        backbone.model = nn.Module()
        backbone.model.embed_tokens = nn.Embedding(20, 128, device="meta")
        backbone.model.layers = nn.ModuleList([FakeExperts(2, device="meta")])
        backbone.model.norm = nn.LayerNorm(128, device="meta")
        backbone.lm_head = nn.Linear(128, 20, device="meta", bias=False)
        backbone.inverse_frequencies = {"unused": SimpleNamespace()}
        placement = materialize_offloaded_backbone(
            backbone, "cpu", embedding_device="cpu", lm_head_device="cpu"
        )
        self.assertEqual(placement["model.layers.0"], "cpu")
        self.assertEqual(backbone.model.layers[0].w13_weight.dtype, FP8_DTYPE)
        self.assertEqual(backbone.model.layers[0].w13_weight.device.type, "cpu")
        self.assertFalse(backbone.inverse_frequencies)
        with self.assertRaisesRegex(ValueError, "fully meta"):
            materialize_offloaded_backbone(backbone, "cpu")


class CudaOffloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        from fish_speech.models.text2semantic.fish_omni import fp8_moe

        if not fp8_moe.fp8_gemm_supported(["cuda:0"]):
            raise unittest.SkipTest("The original FP8 kernel requires Triton and SM89+")
        cls.fp8_moe = fp8_moe

    def reference(self, experts, x, ids, weights):
        device_weights = {
            name: getattr(experts, name).to(x.device) for name in EXPERT_BUFFER_NAMES
        }
        return self.fp8_moe.fp8_moe_forward(
            x, ids, weights, **device_weights
        )

    def test_mixed_and_uncached_paths_match_original_fp8_eight_route_kernel(self):
        experts = FakeExperts(hidden=256, intermediate=128)
        backbone = nn.Module()
        backbone.add_module("experts", experts)
        manager = install_expert_offload(
            backbone,
            backend="triton",
            max_staged_experts=2,
            max_pair_rows=3,
            cache_experts=2,
        )
        self.addCleanup(manager.close)
        x = torch.randn(1, 256, device="cuda:0", dtype=torch.bfloat16)
        weights = torch.softmax(
            torch.randn(1, 8, device="cuda:0"), dim=-1
        ).to(torch.bfloat16)
        ids = torch.tensor(
            [[11, 7, 3, 9, 0, 6, 2, 4]], device="cuda:0"
        )
        expected = self.reference(experts, x, ids, weights)
        for _ in range(2):
            torch.testing.assert_close(
                experts(x, ids, weights), expected, rtol=0, atol=0
            )
        self.assertEqual(manager.stats.hot_cache_hits, 2)
        self.assertEqual(manager.stats.hot_cache_loads, 2)
        self.assertGreater(manager.stats.h2d_bytes, 0)
        manager.configure_hot_cache(0)
        torch.testing.assert_close(
            experts(x, ids, weights), expected, rtol=0, atol=0
        )

    def test_mixed_cache_cross_stream_staging_preserves_dma_sources(self):
        experts = FakeExperts(hidden=256, intermediate=128)
        second = FakeExperts(hidden=256, intermediate=128)
        second.w2_scale_inv.mul_(2)
        backbone = nn.Module()
        backbone.add_module("first", experts)
        backbone.add_module("second", second)
        manager = install_expert_offload(
            backbone,
            backend="triton",
            max_staged_experts=2,
            max_pair_rows=3,
            cache_experts=2,
        )
        self.addCleanup(manager.close)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for index in range(8):
            module = experts if index % 2 == 0 else second
            rows = 1 if index < 4 else 3
            x = torch.randn(rows, 256, device="cuda:0", dtype=torch.bfloat16)
            ids = torch.tensor(
                [[11, 7, 3, 9, 0, 6, 2, 4]] * rows, device="cuda:0"
            )
            weights = torch.softmax(
                torch.randn(rows, 8, device="cuda:0"), dim=-1
            ).to(torch.bfloat16)
            expected = self.reference(module, x, ids, weights)
            stream = streams[index % 2]
            stream.wait_stream(torch.cuda.current_stream())
            for tensor in (x, ids, weights):
                tensor.record_stream(stream)
            with torch.cuda.stream(stream):
                results.append((module(x, ids, weights), expected))
        manager.synchronize()
        for actual, expected in results:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(manager._hot_caches), 2)
        self.assertEqual(manager.stats.hot_cache_loads, 4)
        self.assertGreaterEqual(manager.stats.hot_cache_hits, 12)

    def test_host_tiers_and_warm_reuse_release_match_original_eight_route_kernel(self):
        experts = FakeExperts(hidden=256, intermediate=128)
        backbone = nn.Module()
        backbone.add_module("experts", experts)
        manager = install_expert_offload(
            backbone,
            backend="triton",
            max_staged_experts=2,
            max_pair_rows=3,
            cache_experts=2,
            profile_transfers=True,
        )
        self.addCleanup(manager.close)
        x = torch.randn(1, 256, device="cuda:0", dtype=torch.bfloat16)
        ids = torch.tensor([[11, 7, 3, 9, 0, 6, 2, 4]], device="cuda:0")
        weights = torch.softmax(torch.randn(1, 8, device="cuda:0"), -1).to(torch.bfloat16)
        expected = self.reference(experts, x, ids, weights)
        expert_bytes = ExpertStore.from_module(experts).nbytes // 12
        for name, capacity in (("mapped", 0), ("mixed", 4), ("resident", 12)):
            with self.subTest(tier=name):
                manager.release_runtime_buffers()
                manager.configure_host_residency(capacity * expert_bytes)
                for _ in range(2):
                    torch.testing.assert_close(experts(x, ids, weights), expected, rtol=0, atol=0)
                cache = manager._hot_caches[id(experts)]
                address = cache._device["w13_weight"].data_ptr()
                loads = manager.stats.hot_cache_loads
                manager.configure_hot_cache(2)
                manager.configure_staging(2, 3)
                manager.prepare_buffers("cuda:0")
                torch.testing.assert_close(experts(x, ids, weights), expected, rtol=0, atol=0)
                self.assertEqual(cache._device["w13_weight"].data_ptr(), address)
                self.assertEqual(manager.stats.hot_cache_loads, loads)
                resident_bytes = manager.host_residency_report()["resident_bytes"]
                manager.release_runtime_buffers()
                self.assertEqual(manager.stats.hot_cache_bytes, 0)
                self.assertEqual(manager.stats.device_buffer_bytes, 0)
                self.assertEqual(manager.host_residency_report()["resident_bytes"], resident_bytes)
                manager.prepare_buffers("cuda:0")
                torch.testing.assert_close(experts(x, ids, weights), expected, rtol=0, atol=0)
                manager.synchronize()
                self.assertEqual(manager.stats.hot_cache_loads, loads + 2)
        self.assertGreater(manager.stats.h2d_seconds, 0)
        self.assertGreater(manager.stats.h2d_submit_seconds, 0)
        self.assertGreater(manager.stats.host_gather_seconds, 0)


if __name__ == "__main__":
    unittest.main()
