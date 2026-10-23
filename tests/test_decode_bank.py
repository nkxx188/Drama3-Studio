"""Combined expert banks must retain all routes, limits and layer identities."""

import pytest
import torch
from torch import nn

from drama_low_vram.offload import (
    FP8_DTYPE,
    StagedExperts,
    TorchExpertBackend,
    install_expert_offload,
)


class Experts(nn.Module):
    def __init__(self, *, seed=41):
        super().__init__()
        self.num_experts = 16
        self.fp8 = True
        generator = torch.Generator().manual_seed(seed)
        for name, rows, columns in (
            ("w13", 256, 128),
            ("w2", 128, 128),
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
                    16, rows // 128, columns // 128, generator=generator
                ) * 0.02 + 0.01,
            )

    def forward(self, *args):
        raise AssertionError("Expert manager must intercept forwarding")


class RecordingBackend(TorchExpertBackend):
    def __init__(self):
        self.direct_calls = 0
        self.pair_calls = 0
        self.routed_pairs = 0
        self.bank_counts = []

    def forward(self, x, ids, weights, staged):
        self.direct_calls += 1
        self.routed_pairs += ids.numel()
        self.bank_counts.append(staged.count)
        # Do not route through this class's forward_pairs recorder as well.
        tokens, top_k = ids.shape
        values = TorchExpertBackend.forward_pairs(
            self,
            x.repeat_interleave(top_k, 0),
            ids.reshape(-1, 1),
            weights.reshape(-1, 1),
            staged,
        )
        return values.view(tokens, top_k, x.shape[-1]).sum(1).to(x.dtype)

    def forward_pairs(self, x, ids, weights, staged):
        self.pair_calls += 1
        self.routed_pairs += ids.numel()
        self.bank_counts.append(staged.count)
        return super().forward_pairs(x, ids, weights, staged)


def setup(capacity=8, cache=3, pairs=64, enabled=True):
    backbone = nn.Module()
    experts = Experts()
    backbone.add_module("experts", experts)
    backend = RecordingBackend()
    manager = install_expert_offload(
        backbone,
        backend=backend,
        max_staged_experts=capacity,
        max_pair_rows=pairs,
        cache_experts=cache,
        combined_decode_bank=enabled,
        pin_memory=False,
    )
    return experts, backend, manager


def reference(experts, x, ids, weights):
    store = StagedExperts(
        experts.w13_weight,
        experts.w13_scale_inv,
        experts.w2_weight,
        experts.w2_scale_inv,
    )
    return TorchExpertBackend().forward(x, ids, weights, store)


def test_mixed_decode_becomes_one_forward_and_transfers_only_cold_experts():
    experts, backend, manager = setup()
    x = torch.randn(1, 128)
    warm_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    weights = torch.softmax(torch.randn(1, 8), -1)
    experts(x, warm_ids, weights)
    ids = torch.tensor([[5, 2, 3, 4, 1, 6, 8, 10]])
    before = manager.stats.as_dict()
    direct_before, pair_before = backend.direct_calls, backend.pair_calls
    actual = experts(x, ids, weights)
    torch.testing.assert_close(actual, reference(experts, x, ids, weights), rtol=0, atol=0)
    assert backend.direct_calls == direct_before + 1
    assert backend.pair_calls == pair_before
    assert manager.stats.combined_decode_calls == 1
    assert manager.stats.hot_cache_hits - before["hot_cache_hits"] == 3
    assert manager.stats.expert_loads - before["expert_loads"] == 5
    assert manager.stats.staging_calls - before["staging_calls"] == 1
    assert backend.bank_counts[-1] == 8
    manager.close()


def test_combined_path_adopts_new_hot_experts_without_second_host_copy():
    experts, _, manager = setup()
    x = torch.randn(1, 128)
    weights = torch.full((1, 8), 0.125)
    experts(x, torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]]), weights)
    ids = torch.tensor([[3, 2, 4, 6, 8, 10, 12, 14]])
    before = manager.stats.as_dict()
    actual = experts(x, ids, weights)
    torch.testing.assert_close(actual, reference(experts, x, ids, weights), rtol=0, atol=0)
    assert manager.stats.hot_cache_hits - before["hot_cache_hits"] == 1
    assert manager.stats.hot_cache_loads - before["hot_cache_loads"] == 2
    assert manager.stats.expert_loads - before["expert_loads"] == 7
    assert set(manager._hot_caches[id(experts)]._loaded) == {3, 2, 4}
    manager.close()


def test_duplicate_and_reordered_routes_retain_every_pair():
    experts, backend, manager = setup(cache=2)
    x = torch.randn(1, 128)
    weights = torch.softmax(torch.randn(1, 8), -1)
    experts(x, torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]]), weights)
    ids = torch.tensor([[3, 2, 1, 2, 3, 6, 6, 8]])
    before = backend.routed_pairs
    actual = experts(x, ids, weights)
    torch.testing.assert_close(actual, reference(experts, x, ids, weights), rtol=0, atol=0)
    assert backend.routed_pairs - before == 8
    assert backend.bank_counts[-1] == 5
    assert manager.stats.combined_decode_calls == 1
    manager.close()


@pytest.mark.parametrize("capacity,pairs", [(4, 64), (8, 4), (1, 1)])
def test_small_staging_or_pair_limit_retains_partitioned_execution(capacity, pairs):
    experts, backend, manager = setup(capacity=capacity, pairs=pairs)
    x = torch.randn(1, 128)
    weights = torch.softmax(torch.randn(1, 8), -1)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    experts(x, ids, weights)
    actual = experts(x, ids, weights)
    torch.testing.assert_close(actual, reference(experts, x, ids, weights), rtol=0, atol=0)
    assert manager.stats.combined_decode_calls == 0
    assert manager.stats.peak_staged_experts <= capacity
    assert backend.pair_calls > 0
    manager.close()


def test_flag_is_off_by_default():
    backbone = nn.Module()
    experts = Experts()
    backbone.add_module("experts", experts)
    manager = install_expert_offload(
        backbone, backend="torch", cache_experts=3, pin_memory=False
    )
    assert not manager.combined_decode_bank
    with pytest.raises(TypeError, match="boolean"):
        install_expert_offload(
            nn.Sequential(Experts()), backend="torch", combined_decode_bank="yes"
        )
    manager.close()


def test_prefill_does_not_use_or_repopulate_combined_decode_bank():
    experts, _, manager = setup()
    x = torch.randn(1, 128)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    weights = torch.softmax(torch.randn(1, 8), -1)
    experts(x, ids, weights)
    old_loaded = dict(manager._hot_caches[id(experts)]._loaded)
    batch_ids = torch.tensor([[1, 2, 3, 4], [3, 6, 5, 7]])
    batch_x = torch.randn(2, 128)
    batch_weights = torch.softmax(torch.randn(2, 4), -1)
    actual = experts(batch_x, batch_ids, batch_weights)
    torch.testing.assert_close(
        actual, reference(experts, batch_x, batch_ids, batch_weights), rtol=0, atol=0
    )
    assert manager._hot_caches[id(experts)]._loaded == old_loaded
    assert manager.stats.combined_decode_calls == 0
    manager.close()


def test_repeated_layer_switches_and_eviction_preserve_expert_identity():
    experts, _, manager = setup()
    second = Experts(seed=42)
    manager.attach("second", second)
    x = torch.randn(1, 128)
    weights = torch.softmax(torch.randn(1, 8), -1)
    routes = [
        torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]]),
        torch.tensor([[3, 2, 4, 6, 8, 10, 12, 14]]),
        torch.tensor([[2, 1, 4, 6, 9, 8, 11, 10]]),
    ]
    for _ in range(3):
        for ids in routes:
            for module in (experts, second):
                actual = module(x, ids, weights)
                torch.testing.assert_close(
                    actual, reference(module, x, ids, weights), rtol=0, atol=0
                )
    assert manager.stats.combined_decode_calls >= 8
    manager.close()
