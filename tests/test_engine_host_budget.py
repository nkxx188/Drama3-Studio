"""Engine host plans match selected staging without touching CUDA."""

from unittest.mock import Mock

import torch

from drama_low_vram.autotune import GIB, MIB, HardwareInfo, autotune
from drama_low_vram.engine import DramaEngine, EngineOptions
from drama_low_vram.host_budget import HostMemoryRequirements, estimate_host_requirements


def hardware(available=48):
    return HardwareInfo.from_gib(
        platform="Windows", vram=32, ram=64, ram_available=available,
        commit_available=64, triton_available=True, cpu_logical_cores=16, compute_capability=(8, 9),
    )


def test_actual_automatic_staging_updates_plan_and_runtime_once_metadata_is_read(
    tmp_path, monkeypatch,
):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    estimate = Mock(wraps=estimate_host_requirements)
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware())
    monkeypatch.setattr("drama_low_vram.engine.estimate_host_requirements", estimate)
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    engine.load()
    assert estimate.call_count == 1
    assert engine.runtime.max_staged_experts == 64
    assert loader.call_args.kwargs["max_staged_experts"] == 64
    assert engine.host_plan.requirements.staging_bytes == 64 * 3_146_496
    assert engine.runtime.host_plan.requirements == engine.host_plan.requirements
    assert engine.runtime.host_plan.required_commit_bytes == engine.host_plan.required_commit_bytes
    assert engine.runtime.host_plan.required_physical_bytes == engine.host_plan.required_physical_bytes


def test_final_staging_can_switch_auto_bank_to_mmap_before_loading(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, expert_cache_dir=cache))
    values = {
        name: 0 for name in HostMemoryRequirements.__dataclass_fields__
        if name != "source"
    }
    requirements = HostMemoryRequirements(**{
        **values, "expert_bytes": 20 * GIB, "cpu_weight_bytes": GIB,
        "parking_bytes": GIB, "staging_bytes": 4 * 100 * MIB,
    })
    estimate = Mock(return_value=requirements)
    policy = Mock(wraps=autotune)
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware(24))
    monkeypatch.setattr("drama_low_vram.engine.estimate_host_requirements", estimate)
    monkeypatch.setattr("drama_low_vram.engine.autotune", policy)
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    engine.load()
    assert estimate.call_count == 1
    assert policy.call_count == 2
    assert policy.call_args_list[0].kwargs["expert_storage"] == "ram"
    assert policy.call_args_list[1].kwargs["expert_storage"] == "mmap"
    assert engine.storage == engine.runtime.expert_storage == "mmap"
    assert loader.call_count == 1
    assert loader.call_args.kwargs["expert_cache_dir"] == cache
    assert engine.host_plan.requirements.staging_bytes == 64 * 100 * MIB
    assert engine.host_plan.required_commit_bytes == engine.runtime.required_host_commit_bytes


def test_existing_cpu_reference_codec_is_credited_by_final_staging_plan(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    codec = torch.nn.Linear(8, 4, bias=False)
    codec.register_buffer("mask", torch.ones(4, 4, dtype=torch.bool))
    engine.codec = codec
    codec_bytes = 8 * 4 * 4 + 4 * 4
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware())
    monkeypatch.setattr(
        "drama_low_vram.engine.load_offloaded_drama3",
        Mock(return_value=(object(), object())),
    )
    engine.load()
    assert engine.host_plan.already_allocated_bytes == codec_bytes
    assert engine.runtime.host_plan.already_allocated_bytes == codec_bytes
    assert engine.host_plan.phase_commit_bytes == engine.runtime.host_plan.phase_commit_bytes


def test_explicit_staging_is_preserved_in_estimate_and_loader(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, staged_experts=7))
    estimate = Mock(wraps=estimate_host_requirements)
    policy = Mock(wraps=autotune)
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware())
    monkeypatch.setattr("drama_low_vram.engine.estimate_host_requirements", estimate)
    monkeypatch.setattr("drama_low_vram.engine.autotune", policy)
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    engine.load()
    assert estimate.call_count == policy.call_count == 1
    assert estimate.call_args.kwargs["staged_experts"] == 7
    assert loader.call_args.kwargs["max_staged_experts"] == 7
    assert engine.host_plan.requirements.staging_bytes == 7 * 3_146_496
