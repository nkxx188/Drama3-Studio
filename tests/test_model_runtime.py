"""Resource planning must finish before sampling or cache weights are loaded."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from drama_low_vram.model import _resolve_placement_device, configure_model_runtime
from drama_low_vram.offload import TritonExpertBackend


GIB = 1024**3


class FakeManager:
    def __init__(self, fail_first_allocation=False):
        self.modules = {str(index): object() for index in range(40)}
        self.max_staged_experts = 4
        self.max_pair_rows = 32
        self.backend = object.__new__(TritonExpertBackend)
        self.cache_experts = 0
        self.allocation_attempts = []
        self.fail_first_allocation = fail_first_allocation
        self.stats = SimpleNamespace(device_buffer_bytes=0)
        self.releases = 0

    def release_runtime_buffers(self):
        self.releases += 1
        self.stats.device_buffer_bytes = 0

    def configure_hot_cache(self, capacity):
        self.cache_experts = capacity

    def configure_staging(self, capacity, pairs):
        self.max_staged_experts = capacity
        self.max_pair_rows = pairs

    def prepare_buffers(self, device):
        self.allocation_attempts.append(self.cache_experts)
        if self.fail_first_allocation and len(self.allocation_attempts) == 1:
            raise torch.cuda.OutOfMemoryError("simulated allocation pressure")
        self.stats.device_buffer_bytes = 1


def run_plan(manager, **kwargs):
    model = SimpleNamespace(
        config=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=4096)),
        audio_decoder=SimpleNamespace(device=torch.device("cuda:0")),
        cache_capacity=0,
    )
    store = SimpleNamespace(num_experts=256, nbytes=32220119040 // 40)

    def ensure_caches(model, capacity):
        model.cache_capacity = capacity

    with (
        patch("drama_low_vram.model._require_cuda", return_value=torch.device("cuda:0")),
        patch("drama_low_vram.model.ExpertStore.from_module", return_value=store),
        patch("torch.cuda.device", return_value=nullcontext()),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.empty_cache"),
        patch("torch.cuda.mem_get_info", return_value=(2 * GIB, 8 * GIB)),
        patch("torch.cuda.memory_allocated", return_value=5 * GIB),
        patch("torch.cuda.memory_reserved", return_value=5 * GIB),
        patch(
            "fish_speech.models.text2semantic.fish_omni.generate.ensure_drama3_caches",
            side_effect=ensure_caches,
        ),
    ):
        plan = configure_model_runtime(
            model, manager, sequence_capacity=1024, **kwargs
        )
    return model, plan


def test_automatic_cache_recovers_from_allocation_pressure_before_sampling():
    manager = FakeManager(fail_first_allocation=True)
    model, plan = run_plan(manager)
    assert manager.allocation_attempts == [10, 5]
    assert plan.cache_experts == 5
    assert plan.requested_cache_experts is None
    assert model.cache_capacity == 1024
    assert model._drama_low_vram_allocation["allocation_retries"] == 1
    assert manager.runtime_plan is plan
    assert any("before sampling" in warning for warning in plan.warnings)


def test_manual_cache_allocation_failure_is_not_silently_reduced():
    manager = FakeManager(fail_first_allocation=True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_plan(manager, cache_experts=4)
    assert manager.allocation_attempts == [4]
    assert manager.cache_experts == 0


def test_automatic_codec_headroom_cap_reaches_allocator_and_preserves_oom_recovery():
    runtime = SimpleNamespace(prefill_chunk=32, max_auto_cache_experts=1)
    manager = FakeManager(fail_first_allocation=True)
    _, plan = run_plan(manager, runtime_config=runtime)
    assert manager.allocation_attempts == [1, 0]
    assert plan.cache_experts == 0
    assert plan.requested_cache_experts is None
    manual = FakeManager()
    _, plan = run_plan(manual, runtime_config=runtime, cache_experts=4)
    assert manual.allocation_attempts == [4]
    assert plan.cache_experts == 4


def test_oversized_manual_cache_is_rejected_before_buffer_allocation():
    manager = FakeManager()
    with pytest.raises(RuntimeError, match="short by"):
        run_plan(manager, cache_experts=32, allow_staging_reduction=False)
    assert manager.allocation_attempts == []


def test_manual_batch_and_prefill_limits_are_preserved():
    manager = FakeManager()
    _, plan = run_plan(
        manager, cache_experts=0, prefill_chunk_size=24, allow_staging_reduction=False
    )
    assert plan.cache_experts == 0
    assert plan.prefill_chunk == 24
    assert manager.max_staged_experts == 4
    assert manager.max_pair_rows == 32


class StagingPressure(FakeManager):
    def __init__(self, maximum=1):
        super().__init__()
        self.maximum = maximum
        self.staging_attempts = []

    def prepare_buffers(self, device):
        self.staging_attempts.append(self.max_staged_experts)
        if self.max_staged_experts > self.maximum:
            raise torch.cuda.OutOfMemoryError("shared staging did not fit")
        self.stats.device_buffer_bytes = 1


def test_automatic_allocation_can_shrink_staging_after_cache_is_already_zero():
    manager = StagingPressure()
    runtime = SimpleNamespace(prefill_chunk=32, max_auto_cache_experts=0)
    _, plan = run_plan(manager, runtime_config=runtime)
    assert manager.staging_attempts == [4, 2, 1]
    assert plan.max_staged_experts == 1
    assert plan.max_pair_rows == 8
    assert plan.prefill_chunk == 8
    assert plan.staging_bytes == plan.staged_expert_bytes
    assert plan.can_run and plan.cache_experts == 0


def test_explicit_batch_limits_and_exhausted_minimum_staging_propagate_oom():
    runtime = SimpleNamespace(prefill_chunk=32, max_auto_cache_experts=0)
    manager = StagingPressure()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_plan(manager, runtime_config=runtime, allow_staging_reduction=False)
    assert manager.staging_attempts == [4]
    manager = StagingPressure(maximum=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_plan(manager, runtime_config=runtime)
    assert manager.staging_attempts == [4, 2, 1]


def test_cuda_head_and_embedding_use_the_selected_gpu():
    assert _resolve_placement_device("cuda", torch.device("cuda:2")) == torch.device("cuda:2")
    assert _resolve_placement_device("cpu", torch.device("cuda:2")) == torch.device("cpu")
    assert _resolve_placement_device("cuda:1", torch.device("cuda:2")) == torch.device("cuda:1")
