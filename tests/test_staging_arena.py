"""Packed scratch buffers retain original bytes, routes and allocation limits."""

from contextlib import nullcontext
from unittest.mock import patch
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    ExpertStore,
    FP8_DTYPE,
    OffloadStatistics,
    StagedExperts,
    _StagingPool,
    install_expert_offload,
)


class Experts(nn.Module):
    def __init__(self, *, hidden=128, intermediate=128):
        super().__init__()
        self.fp8 = True
        generator = torch.Generator().manual_seed(20261015)
        for name, rows, columns in (
            ("w13", 2 * intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
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

    def forward(self, *args):
        raise AssertionError("Offload forwarding was not installed")


def create_manager(**options):
    backbone = nn.Sequential(Experts())
    manager = install_expert_offload(
        backbone, backend="torch", pin_memory=False, **options
    )
    return backbone[0], manager


@pytest.mark.parametrize("hidden", [128, 256])
@pytest.mark.parametrize("count", [1, 4, 8])
def test_arena_views_share_bounded_storage_and_preserve_every_role(hidden, count):
    experts = Experts(hidden=hidden)
    store = ExpertStore.from_module(experts)
    stats = OffloadStatistics()
    pool = _StagingPool(8, False, False, stats, packed_staging=True)
    ids = torch.tensor([11, 0, 7, 3, 5, 13, 1, 9][:count])
    original_pointers = {
        name: getattr(experts, name).data_ptr() for name in EXPERT_BUFFER_NAMES
    }
    with pool.session(store, torch.device("cpu")):
        staged = pool.stage(store, ids)
        expert_bytes = store.nbytes // store.num_experts
        assert pool._host_arena.shape == (8, expert_bytes)
        assert pool._device_arena.shape == (8, expert_bytes)
        assert stats.device_buffer_bytes == 8 * expert_bytes
        for buffers, arena in (
            (pool._host, pool._host_arena),
            (pool._device, pool._device_arena),
        ):
            offset = 0
            for name in EXPERT_BUFFER_NAMES:
                tensor = buffers[name]
                source = getattr(store, name)
                assert tensor.data_ptr() == arena.data_ptr() + offset
                assert tensor.untyped_storage().data_ptr() == arena.data_ptr()
                assert tensor[0].is_contiguous()
                assert tensor.stride(0) * tensor.element_size() == expert_bytes
                expected = source.view(torch.uint8).index_select(0, ids)
                assert torch.equal(
                    getattr(staged, name).view(torch.uint8), expected
                )
                offset += source[0].numel() * source.element_size()
        assert stats.packed_staging_calls == stats.staging_calls == 1
        assert stats.h2d_copy_calls == stats.h2d_bytes == 0
        assert stats.staged_bytes == count * expert_bytes
        pool.stage(store, ids.clone())
        assert stats.packed_staging_calls == 1
        assert stats.expert_cache_hits == count
    assert original_pointers == {
        name: getattr(experts, name).data_ptr() for name in EXPERT_BUFFER_NAMES
    }
    pool.close()
    assert pool._host_arena is pool._device_arena is None
    assert not pool._host and not pool._device
    assert stats.device_buffer_bytes == 0


def test_combining_resident_experts_writes_correct_arena_offsets():
    store = ExpertStore.from_module(Experts())
    pool = _StagingPool(
        8, False, False, OffloadStatistics(), packed_staging=True
    )
    cold_ids = torch.tensor([13, 2, 7])
    resident_ids = torch.tensor([1, 3, 5, 9])
    resident = StagedExperts(
        **{
            name: getattr(store, name).view(torch.uint8)
            .index_select(0, resident_ids)
            .view(getattr(store, name).dtype)
            for name in EXPERT_BUFFER_NAMES
        }
    )
    with pool.session(store, torch.device("cpu")):
        pool.stage(store, cold_ids)
        combined = pool.combine_resident(3, resident, [3, 1])
        expected_ids = torch.tensor([13, 2, 7, 9, 3])
        for name in EXPERT_BUFFER_NAMES:
            expected = getattr(store, name).view(torch.uint8).index_select(0, expected_ids)
            assert torch.equal(getattr(combined, name).view(torch.uint8), expected)
    pool.close()


@pytest.mark.parametrize(
    "capacity,cache,combined", [(8, 0, False), (8, 2, True), (4, 2, False)]
)
def test_packed_routes_match_existing_layout_with_cache_and_partitioning(
    capacity, cache, combined
):
    plain, plain_manager = create_manager(
        max_staged_experts=capacity,
        cache_experts=cache,
        combined_decode_bank=combined,
    )
    packed, packed_manager = create_manager(
        max_staged_experts=capacity,
        cache_experts=cache,
        combined_decode_bank=combined,
        packed_staging=True,
    )
    generator = torch.Generator().manual_seed(411)
    x = torch.randn(1, 128, generator=generator)
    weights = torch.softmax(torch.randn(1, 8, generator=generator), dim=-1)
    for route in (
        [1, 3, 5, 7, 9, 11, 13, 15],
        [3, 2, 1, 2, 3, 6, 6, 8],
        [1, 3, 5, 7, 9, 11, 13, 15],
    ):
        ids = torch.tensor([route])
        torch.testing.assert_close(
            packed(x, ids, weights), plain(x, ids, weights), rtol=0, atol=0
        )
    assert packed_manager.stats.packed_staging_calls > 0
    assert plain_manager.stats.packed_staging_calls == 0
    if combined:
        assert packed_manager.stats.combined_decode_calls > 0
        cache_buffers = packed_manager._hot_caches[id(packed)]._device
        assert all(tensor.is_contiguous() for tensor in cache_buffers.values())
    plain_manager.close()
    packed_manager.close()


def test_staging_resize_and_release_preserve_flag_and_recreate_arena():
    experts, manager = create_manager(max_staged_experts=8, packed_staging=True)
    manager.prepare_buffers("cpu")
    arena = manager.pool._device_arena
    manager.configure_staging(8, 32)
    assert manager.pool._device_arena is arena
    manager.configure_staging(4, 32)
    assert manager.packed_staging and manager.pool.packed_staging
    assert manager.pool._device_arena is None
    manager.prepare_buffers("cpu")
    assert manager.pool._device_arena is not arena
    assert manager.pool._device_arena.shape[0] == 4
    manager.release_runtime_buffers()
    assert manager.pool._host_arena is manager.pool._device_arena is None
    assert manager.pool.packed_staging
    manager.prepare_buffers("cpu")
    assert manager.pool._device_arena.shape[0] == 4
    manager.close()


def test_packed_pinned_allocation_failure_uses_one_pageable_arena():
    experts, manager = create_manager(max_staged_experts=8, packed_staging=True)
    manager.pool.pin_memory = True
    real_empty = torch.empty
    calls = []

    def allocate(*args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("pin_memory"):
            raise RuntimeError("synthetic Windows pinned host-lock allocation limit")
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    with patch("drama_low_vram.offload.torch.empty", side_effect=allocate):
        manager.pool._allocate(ExpertStore.from_module(experts), torch.device("cuda:0"))
    assert len(calls) == 3
    assert manager.stats.pinned_allocation_fallbacks == 1
    assert not manager.pool.pin_memory
    assert not manager.pool._host_arena.is_pinned()
    assert manager.stats.pinned_bytes == 0
    manager.close()


def test_packed_device_oom_leaves_no_live_arena():
    experts, manager = create_manager(max_staged_experts=8, packed_staging=True)
    real_empty = torch.empty

    def allocate(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            raise torch.OutOfMemoryError("synthetic GPU arena allocation failure")
        return real_empty(*args, **kwargs)

    with patch("drama_low_vram.offload.torch.empty", side_effect=allocate):
        with pytest.raises(torch.OutOfMemoryError, match="GPU arena"):
            manager.pool._allocate(
                ExpertStore.from_module(experts), torch.device("cuda:0")
            )
    assert manager.pool._host_arena is manager.pool._device_arena is None
    assert not manager.pool._host and not manager.pool._device
    assert manager.pool._target is None
    assert manager.stats.pinned_allocation_fallbacks == 0
    manager.close()


def test_packed_staging_flag_is_explicit_and_requires_a_boolean():
    _, manager = create_manager()
    assert not manager.packed_staging
    assert not manager.pool.packed_staging
    manager.close()
    with pytest.raises(TypeError, match="packed_staging must be a boolean"):
        create_manager(packed_staging="yes")


@pytest.mark.parametrize("packed,expected_copy_calls", [(False, 4), (True, 1)])
def test_cuda_copy_submission_accounting_uses_one_arena_without_a_gpu(
    packed, expected_copy_calls
):
    store = ExpertStore.from_module(Experts())
    stats = OffloadStatistics()
    pool = _StagingPool(8, False, False, stats, packed_staging=packed)
    pool._allocate(store, torch.device("cpu"))
    # CPU tensors plus mocked events exercise submission accounting without
    # allocating a CUDA tensor or initializing a driver.
    pool._target = torch.device("cuda:0")
    copy_calls = []
    original_copy = torch.Tensor.copy_

    def copy(target, source, *, non_blocking=False):
        copy_calls.append((target.shape, source.shape))
        return original_copy(target, source, non_blocking=non_blocking)

    ids = torch.tensor([13, 2, 7])
    with (
        patch.object(torch.Tensor, "copy_", new=copy),
        patch("drama_low_vram.offload.torch.cuda.device", return_value=nullcontext()),
        patch("drama_low_vram.offload.torch.cuda.Event", return_value=Mock()),
        patch("drama_low_vram.offload.torch.cuda.current_stream", return_value=object()),
    ):
        staged = pool.stage(store, ids)
        assert len(copy_calls) == stats.h2d_copy_calls == expected_copy_calls
        assert stats.h2d_bytes == 3 * store.nbytes // store.num_experts
        assert stats.packed_staging_calls == int(packed)
        for name in EXPERT_BUFFER_NAMES:
            assert torch.equal(
                getattr(staged, name).view(torch.uint8),
                getattr(store, name).view(torch.uint8).index_select(0, ids),
            )
        pool.stage(store, ids)
        assert len(copy_calls) == stats.h2d_copy_calls == expected_copy_calls
        pool.close()
