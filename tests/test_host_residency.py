"""Host residency preserves source bytes while enforcing a stable RAM budget."""

import pytest
import torch

from drama_low_vram.host_residency import HostResidency
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    ExpertStore,
    OffloadStatistics,
    _StagingPool,
)


def make_store(seed=13, experts=5):
    generator = torch.Generator().manual_seed(seed)
    return ExpertStore(
        w13_weight=torch.randint(-4, 5, (experts, 256, 128), generator=generator).to(FP8_DTYPE),
        w13_scale_inv=torch.rand(experts, 2, 1, generator=generator),
        w2_weight=torch.randint(-4, 5, (experts, 128, 128), generator=generator).to(FP8_DTYPE),
        w2_scale_inv=torch.rand(experts, 1, 1, generator=generator),
    )


def buffers(store, count, *, packed=False):
    if packed:
        pool = _StagingPool(count, False, False, OffloadStatistics(), packed_staging=True)
        pool._allocate(store, torch.device("cpu"))
        return pool._host
    return {
        name: torch.empty((count, *getattr(store, name).shape[1:]), dtype=getattr(store, name).dtype)
        for name in EXPERT_BUFFER_NAMES
    }


def equal_bytes(left, right):
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def test_default_rank_spreads_unseen_experts_across_layers():
    host = HostResidency({2: make_store(), 4: make_store(14, 3)})
    assert host.ranked_keys()[:6] == ((2, 0), (4, 0), (2, 1), (4, 1), (2, 2), (4, 2))
    host.configure(host.expert_bytes * 3 + 5)
    assert host.resident_keys == ((2, 0), (4, 0), (2, 1))
    assert host.resident_bytes <= host.budget_bytes


def test_rank_and_gpu_complement_exclusions_are_bounded():
    host = HostResidency([make_store(), make_store(14)])
    rank = [(1, 3), (0, 2), (1, 3), (0, 1), (1, 4)]
    report = host.configure(host.expert_bytes * 2, rank, excluded_keys=[(1, 3)])
    assert host.resident_keys == ((0, 2), (0, 1))
    assert report['resident_experts'] == 2
    assert report['copied_bytes'] == host.expert_bytes * 2
    host.set_budget(host.total_source_bytes)
    assert (1, 3) not in host.resident_keys
    assert len(host.resident_keys) == host.total_experts - 1


@pytest.mark.parametrize('keys', [(), ((0, 0),), ((0, 1), (0, 3)), tuple((0, i) for i in range(5))])
@pytest.mark.parametrize('packed', [False, True])
def test_gather_matches_original_bytes_for_all_tier_patterns_and_duplicates(keys, packed):
    store = make_store()
    host = HostResidency([store])
    host.configure(host.expert_bytes * len(keys), keys)
    ids = torch.tensor([3, 1, 1, 0, 4, 2, 3], dtype=torch.int64)
    targets = buffers(store, len(ids) + 1, packed=packed)
    for target in targets.values():
        target[-1].view(torch.uint8).fill_(231)
    host.copy_to(0, ids, targets)
    for name in EXPERT_BUFFER_NAMES:
        source = getattr(store, name)
        assert equal_bytes(targets[name][:-1], source.view(torch.uint8).index_select(0, ids).view(source.dtype))
        assert torch.all(targets[name][-1].view(torch.uint8) == 231)
    expected_hits = sum((0, i) in keys for i in ids.tolist())
    assert host.stats.resident_hits == expected_hits
    assert host.stats.mapped_fallback_reads == len(ids) - expected_hits


def test_resident_storage_is_an_independent_copy_and_cold_storage_is_original_view():
    store = make_store()
    host = HostResidency([store])
    host.configure(host.expert_bytes, [(0, 1)])
    hot = host.get(0, 1)
    cold = host.get(0, 2)
    for name in EXPERT_BUFFER_NAMES:
        source = getattr(store, name)
        assert equal_bytes(hot[name], source[1])
        assert hot[name].untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        assert cold[name].untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


def test_cold_reads_only_observe_and_explicit_boundary_promotes_by_frequency():
    host = HostResidency([make_store(), make_store(14)])
    host.configure(host.expert_bytes, [(0, 0)])
    for _ in range(4):
        host.get(1, 3)
    host.get(0, 2)
    assert host.resident_keys == ((0, 0),)
    assert host.ranked_keys(include_unseen=False) == ((1, 3), (0, 2))
    host.configure(host.expert_bytes * 2)
    assert host.resident_keys == ((1, 3), (0, 2))


def test_full_router_observation_does_not_count_source_reads_or_double_count_staging():
    store = make_store()
    host = HostResidency([store])
    host.observe(0, torch.tensor([[4, 1, 1], [2, 1, 4]]))
    assert host.ranked_keys(include_unseen=False) == ((0, 1), (0, 4), (0, 2))
    assert host.stats.observed_routes == 6
    assert host.stats.expert_reads == 0
    host.copy_to(0, torch.tensor([2, 4]), buffers(store, 2), observe=False)
    assert host.stats.observed_routes == 6
    assert host.stats.expert_reads == 2
    assert host.ranked_keys(include_unseen=False) == ((0, 1), (0, 4), (0, 2))


@pytest.mark.parametrize('ids', [torch.tensor([]), torch.tensor([0.5]), torch.tensor([[[0]]]), torch.tensor([True]), torch.tensor([9])])
def test_invalid_observations_are_rejected_without_partial_counts(ids):
    host = HostResidency([make_store()])
    with pytest.raises((ValueError, IndexError)):
        host.observe(0, ids)
    assert host.stats.observed_routes == 0


def test_no_change_does_not_rebuild_or_reallocate():
    host = HostResidency([make_store()])
    host.configure(host.expert_bytes, [(0, 1)])
    pointer = host.get(0, 1)['w13_weight'].data_ptr()
    host.configure(host.expert_bytes + 1, [(0, 1)])
    assert host.get(0, 1)['w13_weight'].data_ptr() == pointer
    assert host.stats.rebuilds == 1


def test_budget_shrink_releases_excess_and_zero_keeps_fallback():
    host = HostResidency([make_store()])
    host.configure(host.expert_bytes * 4)
    host.set_budget(host.expert_bytes * 2)
    assert host.resident_keys == ((0, 0), (0, 1))
    assert host.resident_bytes == host.expert_bytes * 2
    host.set_budget(0)
    assert host.resident_keys == ()
    assert host._arena == {}
    assert not host.enabled
    host.get(0, 0)
    assert host.stats.mapped_fallback_reads == 1


def test_bad_rank_does_not_destroy_existing_configuration():
    host = HostResidency([make_store()])
    host.configure(host.expert_bytes, [(0, 1)])
    with pytest.raises(IndexError):
        host.configure(host.expert_bytes * 2, [(0, 2), (9, 0)])
    assert host.resident_keys == ((0, 1),)
    assert host.budget_bytes == host.expert_bytes


def test_allocation_failure_falls_back_without_partial_publication(monkeypatch):
    host = HostResidency([make_store()])
    host.configure(host.expert_bytes, [(0, 1)])
    original = torch.empty
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('DefaultCPUAllocator: not enough memory')
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', fail_second)
    result = host.configure(host.expert_bytes * 2)
    assert result['allocation_fallbacks'] == 1
    assert host.resident_bytes == 0
    assert host._arena == {}
    assert 'not enough memory' in host.last_error
    assert equal_bytes(host.get(0, 4)['w2_weight'], host._stores[0].w2_weight[4])


def test_reclaim_only_receives_copied_keys_and_failure_preserves_arena():
    reclaimed = []

    def reclaim(keys):
        reclaimed.append(keys)
        raise OSError('unsupported platform')

    host = HostResidency([make_store()], reclaim_copied=reclaim)
    host.configure(host.expert_bytes * 2, [(0, 4), (0, 2)])
    assert reclaimed == [((0, 4), (0, 2))]
    assert host.resident_keys == ((0, 4), (0, 2))
    assert host.stats.reclaim_failures == 1


@pytest.mark.parametrize('budget', [-1, True, 1.5, None])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        HostResidency([make_store()]).configure(budget)


@pytest.mark.parametrize('ids', [torch.tensor([], dtype=torch.int64), torch.tensor([0.0]), torch.tensor([[0]]), torch.tensor([8])])
def test_invalid_ids_do_not_record_or_write(ids):
    store = make_store()
    host = HostResidency([store])
    with pytest.raises((ValueError, IndexError)):
        host.copy_to(0, ids, buffers(store, 2))
    assert host.stats.requests == 0


def test_invalid_output_does_not_partially_gather():
    store = make_store()
    host = HostResidency([store])
    target = buffers(store, 2)
    target['w13_weight'].view(torch.uint8).fill_(79)
    del target['w2_weight']
    with pytest.raises(ValueError):
        host.copy_to(0, torch.tensor([0, 1]), target)
    assert torch.all(target['w13_weight'].view(torch.uint8) == 79)
    assert host.stats.requests == 0


@pytest.mark.parametrize('layout', ['transposed', 'overlapping'])
def test_invalid_output_layout_does_not_partially_gather(layout):
    store = make_store()
    host = HostResidency([store])
    target = buffers(store, 2)
    target['w13_weight'].view(torch.uint8).fill_(79)
    original = target['w2_weight']
    if layout == 'transposed':
        target['w2_weight'] = original.transpose(1, 2)
    else:
        target['w2_weight'] = original[:1].expand_as(original)
    with pytest.raises(ValueError):
        host.copy_to(0, torch.tensor([0, 1]), target)
    assert torch.all(target['w13_weight'].view(torch.uint8) == 79)
    assert host.stats.requests == 0


def test_close_is_idempotent_and_releases_source_ownership():
    host = HostResidency([make_store()])
    host.configure(host.expert_bytes)
    host.close()
    host.close()
    assert host._arena == host._stores == {}
    assert host.resident_bytes == 0
    for call in [lambda: host.get(0, 0), lambda: host.configure(1), lambda: host.ranked_keys(), lambda: host.set_budget(1)]:
        with pytest.raises(RuntimeError, match='closed'):
            call()


def test_uniform_geometry_and_nonempty_layers_are_required():
    with pytest.raises(ValueError, match='at least one'):
        HostResidency([])
    with pytest.raises(ValueError, match='non-negative'):
        HostResidency({True: make_store()})
    different = make_store()
    different = ExpertStore(
        w13_weight=different.w13_weight[:, :128],
        w13_scale_inv=different.w13_scale_inv[:, :1],
        w2_weight=different.w2_weight[:, :, :64],
        w2_scale_inv=different.w2_scale_inv,
    )
    with pytest.raises(ValueError):
        HostResidency([make_store(), different])
