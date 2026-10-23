"""Future allocation peaks, file-backed experts, and Windows commit policy."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from drama_low_vram.host_budget import (
    GIB,
    MIB,
    HostAllocationState,
    HostMemoryRequirements,
    estimate_host_requirements,
    plan_host_memory,
)


def requirements(**overrides):
    values = {
        name: 0 for name in HostMemoryRequirements.__dataclass_fields__
        if name != "source"
    }
    values.update(overrides)
    return HostMemoryRequirements(**values)


def plan(req, *, ram=1000, commit=1000, storage="auto", allocated=None):
    return plan_host_memory(
        ram_available_bytes=ram,
        commit_available_bytes=commit,
        expert_storage=storage,
        requirements=req,
        allocated=allocated,
        buffer_bytes=0,
    )


def test_auto_prefers_all_experts_in_ram_at_the_exact_future_peak():
    req = requirements(expert_bytes=600, cpu_weight_bytes=100, parking_bytes=100)
    exact = plan(req, ram=800, commit=800)
    assert exact.expert_storage == "ram"
    assert exact.required_physical_bytes == 800
    assert exact.required_commit_bytes == 800
    assert exact.expert_ram_budget_bytes == 600
    assert exact.can_run
    assert plan(req, ram=799).expert_storage == "mmap"
    assert plan(req, commit=799).expert_storage == "mmap"


def test_available_ram_does_not_have_a_second_fixed_system_reserve():
    req = requirements(expert_bytes=30 * GIB, cpu_weight_bytes=GIB)
    result = plan_host_memory(
        ram_available_bytes=32 * GIB,
        commit_available_bytes=32 * GIB,
        requirements=req,
    )
    assert result.expert_storage == "ram"
    assert result.required_physical_bytes == 31 * GIB + result.buffer_bytes
    assert 256 * MIB <= result.buffer_bytes <= 512 * MIB
    assert result.buffer_bytes < GIB


def test_model_loading_and_codec_constructor_are_alternative_peaks():
    req = requirements(
        expert_bytes=500, cpu_weight_bytes=100,
        ram_load_temporary_bytes=200, load_file_window_bytes=50,
        parking_bytes=100, codec_resident_bytes=40, codec_build_bytes=150,
        codec_load_temporary_bytes=20, codec_checkpoint_commit_bytes=300,
        codec_file_window_bytes=30,
    )
    result = plan(req, storage="ram")
    assert result.phase_physical_bytes["model_load"] == 850
    assert result.phase_physical_bytes["codec_load"] == 800
    assert result.phase_commit_bytes["codec_load"] == 1070
    assert result.required_physical_bytes == 850
    assert result.required_commit_bytes == 1070
    assert not result.can_run


def test_readonly_expert_pages_do_not_charge_private_commit():
    req = requirements(
        expert_bytes=30 * GIB, cpu_weight_bytes=GIB,
        load_file_window_bytes=GIB, parking_bytes=2 * GIB,
    )
    result = plan(req, ram=4 * GIB, commit=3 * GIB)
    assert result.expert_storage == "mmap"
    assert result.can_run
    assert result.required_commit_bytes == 3 * GIB
    assert result.expert_ram_budget_bytes == 0


def test_actual_commit_is_never_added_to_physical_ram_again():
    req = requirements(cpu_weight_bytes=200)
    result = plan_host_memory(
        ram_available_bytes=1000,
        commit_available_bytes=100,
        swap_available_bytes=10000,
        requirements=req,
        buffer_bytes=0,
    )
    assert result.available_commit_bytes == 100
    assert not result.commit_budget_estimated
    assert not result.can_run
    estimated = plan_host_memory(
        ram_available_bytes=100,
        swap_available_bytes=150,
        requirements=req,
        buffer_bytes=0,
    )
    assert estimated.available_commit_bytes == 250
    assert estimated.commit_budget_estimated
    assert estimated.can_run


def test_preloaded_reference_codec_is_credited_without_rebuilding_it():
    req = requirements(
        expert_bytes=600, cpu_weight_bytes=100, parking_bytes=100,
        codec_resident_bytes=80, codec_build_bytes=200,
        codec_load_temporary_bytes=50, codec_checkpoint_commit_bytes=300,
    )
    cold = plan(req, storage="ram", commit=2000)
    warm = plan(
        req, ram=920, commit=920, storage="ram",
        allocated=HostAllocationState(codec_bytes=80),
    )
    assert cold.required_commit_bytes == 1250
    assert warm.already_allocated_bytes == 80
    assert warm.required_commit_bytes == 800
    assert warm.phase_commit_bytes["model_load"] == 700
    assert warm.can_run
    assert warm.required_commit_bytes < cold.required_commit_bytes - 80


def test_codec_construction_keeps_generation_staging_but_not_future_parking():
    req = requirements(
        cpu_weight_bytes=100, parking_bytes=300, staging_bytes=70,
        codec_build_bytes=500, codec_load_temporary_bytes=50,
    )
    result = plan(req)
    assert result.phase_commit_bytes["codec_load"] == 720
    already_parked = plan(req, allocated=HostAllocationState(parking_bytes=300))
    # Existing mirrors are already charged to the availability snapshot.
    assert already_parked.phase_commit_bytes["codec_load"] == 720


def test_preloaded_smaller_codec_retains_only_growth_scratch():
    req = requirements(
        cpu_weight_bytes=100, parking_bytes=100, staging_bytes=10,
        codec_resident_bytes=200, codec_build_bytes=1000,
        codec_load_temporary_bytes=50, codec_checkpoint_commit_bytes=1000,
    )
    warm = plan(req, allocated=HostAllocationState(codec_bytes=80))
    assert warm.phase_commit_bytes["codec_load"] == 380
    assert warm.phase_physical_bytes["codec_load"] == 380
    complete = plan(req, allocated=HostAllocationState(codec_bytes=200))
    assert complete.phase_commit_bytes["codec_load"] == 210


def test_existing_parking_and_transfer_buffers_are_not_reserved_twice():
    req = requirements(
        expert_bytes=600, cpu_weight_bytes=100,
        parking_bytes=100, staging_bytes=20,
    )
    before = plan(req, storage="ram")
    after = plan(
        req, storage="ram",
        allocated=HostAllocationState(parking_bytes=100, staging_bytes=20),
    )
    assert after.already_allocated_bytes == 120
    assert after.required_commit_bytes == 700
    assert after.phase_commit_bytes["generation"] == 700
    # The existing mirrors are present in phases where a cold engine has none.
    assert before.required_commit_bytes == 820


def test_physical_pressure_uses_mmap_and_warns_while_commit_allows_paging():
    req = requirements(expert_bytes=600, cpu_weight_bytes=100, parking_bytes=300)
    result = plan(req, ram=200, commit=1000)
    assert result.expert_storage == "mmap"
    assert not result.physical_fits
    assert result.commit_fits
    assert result.can_run
    assert any("very slow" in warning for warning in result.warnings)
    explicit = plan(req, ram=200, commit=1000, storage="ram")
    assert explicit.expert_storage == "ram"
    assert explicit.can_run


def test_resident_expert_budget_uses_both_host_limits():
    req = requirements(expert_bytes=600, cpu_weight_bytes=100, parking_bytes=100)
    assert plan(req, ram=500, commit=1000).expert_ram_budget_bytes == 300
    assert plan(req, ram=1000, commit=500).expert_ram_budget_bytes == 300
    assert plan(req, ram=1000, commit=200).expert_ram_budget_bytes == 0


def test_default_estimate_matches_measured_official_checkpoint_dimensions():
    req = estimate_host_requirements()
    assert req == replace(HostMemoryRequirements())
    assert req.expert_bytes == 32_220_119_040
    assert req.parking_bytes == 5_308_033_280 + 16_814_088
    assert req.staging_bytes == 4 * 3_146_496
    assert req.codec_resident_bytes == 782_796_770 + 3 * (4096**2 + 4096 * 64 * 2)
    direct = estimate_host_requirements(direct_compact_head=True, park_model=False)
    assert direct.ram_load_temporary_bytes == req.mmap_load_temporary_bytes
    assert direct.parking_bytes == 0
    no_codec = estimate_host_requirements(include_codec=False)
    assert no_codec.parking_bytes == 0
    assert no_codec.codec_resident_bytes == 0
    assert no_codec.codec_checkpoint_commit_bytes == 0


def test_checkpoint_projection_uses_tensor_metadata_and_placement(tmp_path):
    save_file({
        "model.language_model.embed_tokens.weight": torch.zeros(20, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(20, 4, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.experts.0.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
        "model.language_model.layers.0.mlp.experts.0.weight_scale": torch.ones(4),
        "model.language_model.layers.0.norm.weight": torch.ones(4),
        "audio_decoder.weight": torch.ones(2, 4),
    }, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "semantic_start_token_id": 10,
        "semantic_end_token_id": 12,
        "text_config": {"num_hidden_layers": 1, "num_experts": 1},
    }), encoding="utf-8")
    torch.save({"state_dict": {
        "generator.conv.parametrizations.weight.original0": torch.ones(2, 1),
        "generator.conv.parametrizations.weight.original1": torch.ones(2, 4),
        "generator.transformer.causal_mask": torch.ones(8, 8, dtype=torch.bool),
        "generator.transformer.freqs_cis": torch.ones(8, 2, 2),
    }}, tmp_path / "codec.pth")
    with patch.object(
        torch.UntypedStorage, "from_file",
        side_effect=AssertionError("Metadata inspection must not map payloads"),
    ):
        req = estimate_host_requirements(
            tmp_path, staged_experts=2, codec_max_frames=2,
            embedding_device="cuda", direct_compact_head=True,
        )
    assert req.expert_bytes == 32
    assert req.cpu_weight_bytes == 0
    assert req.parking_bytes == 24 + 160 + 64
    assert req.fixed_gpu_bytes == 24 + 160 + 64
    assert req.num_experts_per_layer == 1
    assert req.staging_bytes == 64
    assert req.ram_load_temporary_bytes == 64
    assert req.load_file_window_bytes == 160
    assert req.codec_resident_bytes == 16 + 64 + 8 * 4 * 2
    assert req.codec_build_bytes == 40 + 64 + 8 * 4 * 2
    assert req.codec_load_temporary_bytes == 32 + 64
    assert req.codec_checkpoint_commit_bytes == (tmp_path / "codec.pth").stat().st_size
    assert req.source == str(tmp_path.resolve())


def test_plan_is_json_serializable():
    result = plan(requirements(expert_bytes=600))
    payload = json.loads(result.to_json())
    assert payload["can_run"]
    assert payload["requirements"]["expert_bytes"] == 600
    assert isinstance(payload["warnings"], list)


def test_shared_bf16_sources_are_not_counted_as_private_cpu_copies(tmp_path):
    packed = tmp_path / "packed.safetensors"
    save_file({
        "model.embed_tokens.weight": torch.ones(20, 4, dtype=torch.bfloat16),
        "audio_head.weight": torch.ones(4, 4, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.gate_up_weight": torch.zeros(1, 8, 4, dtype=torch.float8_e4m3fn),
        "model.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "tower.norm.weight": torch.ones(4, dtype=torch.float32),
    }, str(packed))
    (tmp_path / "shared-model.json").write_text(json.dumps({
        "format": "drama3-packed-shared-v1", "model": str(packed),
        "size": packed.stat().st_size, "sha256": "a" * 64,
    }), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({
        "semantic_start_token_id": 10, "semantic_end_token_id": 12,
        "text_config": {"num_hidden_layers": 1, "num_experts": 1},
    }), encoding="utf-8")
    # Codec metadata is independent of the new weight-source accounting.
    torch.save({"generator.weight": torch.ones(2, 4)}, tmp_path / "codec.pth")
    req = estimate_host_requirements(tmp_path, codec_max_frames=2)
    assert req.cpu_weight_bytes == 0
    assert req.file_backed_cpu_weight_bytes == 20 * 4 * 2
    assert req.file_backed_parking_bytes == 4 * 2 + 4 * 4 * 2
    # The FP32 source needs BF16 conversion, while candidate IDs are generated.
    assert req.parking_bytes == 4 * 2 + 4 * 8


@pytest.mark.parametrize("quantization", ["int8_convrot", "w6a8", "w4a8"])
def test_comfy_estimates_the_staged_format_without_json_or_duplicate_codebooks(tmp_path, quantization):
    tensors = {
        "model.embed_tokens.weight": torch.zeros(20, 512, dtype=torch.bfloat16),
        "audio_head.weight": torch.zeros(4, 512, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(512, dtype=torch.bfloat16),
        "tower.norm.weight": torch.ones(512, dtype=torch.bfloat16),
    }
    expected_bank, expected_aux = 0, 0
    for projection, rows in (("gate_up_proj", 1024), ("down_proj", 512)):
        base = "model.layers.0.mlp.experts." + projection
        columns = 512 if quantization == "int8_convrot" else 384 if quantization == "w6a8" else 256
        tensors[base + ".weight"] = torch.zeros(2, rows, columns, dtype=torch.int8)
        expected_bank += 2 * rows * columns
        tensors[base + ".comfy_quant"] = torch.zeros(200, dtype=torch.uint8)
        if quantization == "int8_convrot":
            tensors[base + ".weight_scale"] = torch.ones(2, rows)
            expected_bank += 2 * rows * 4
        else:
            tensors[base + ".weight_s_rel"] = torch.ones(2, rows, 32, dtype=torch.uint8)
            tensors[base + ".weight_s_channel"] = torch.ones(2, rows)
            expected_aux += 2 * rows * (36 if quantization == "w6a8" else 40)
            if quantization == "w4a8":
                tensors[base + ".weight_codebook"] = torch.arange(16, dtype=torch.float32)
    packed = tmp_path / "packed.safetensors"
    save_file(tensors, str(packed))
    (tmp_path / "shared-model.json").write_text(json.dumps({
        "format": "drama3-packed-shared-v1", "model": str(packed),
        "size": packed.stat().st_size, "sha256": "a" * 64, "quantization": quantization,
    }))
    (tmp_path / "config.json").write_text(json.dumps({
        "semantic_start_token_id": 10, "semantic_end_token_id": 12,
        "text_config": {"num_hidden_layers": 1, "num_experts": 2},
    }))
    with patch.object(torch.UntypedStorage, "from_file", side_effect=AssertionError("payload read")):
        req = estimate_host_requirements(tmp_path, staged_experts=2, include_codec=False)
    assert req.expert_bytes == expected_bank + expected_aux
    assert req.staging_bytes == req.expert_bytes
    assert req.cpu_weight_bytes == 20 * 512 * 2 + expected_aux
    assert req.fixed_gpu_bytes == 2 * 512 * 2 + 4 * 512 * 2 + 4 * 8


def test_gptq_gpu_estimate_keeps_compressed_dense_weights_and_excludes_validation_tensors(tmp_path):
    prefix = "model.language_model.layers.0"
    save_file({
        "model.language_model.embed_tokens.weight": torch.zeros(20, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(20, 4, dtype=torch.bfloat16),
        prefix + ".dense.qweight": torch.zeros(1, 4, dtype=torch.int32),
        prefix + ".dense.scales": torch.ones(1, 4, dtype=torch.float16),
        prefix + ".dense.qzeros": torch.ones(1, 1, dtype=torch.int32),
        prefix + ".dense.g_idx": torch.ones(8, dtype=torch.int32),
        prefix + ".norm.weight": torch.ones(4),
        prefix + ".mlp.experts.0.qweight": torch.zeros(2, 1, 4, dtype=torch.int32),
        prefix + ".mlp.experts.0.scales": torch.ones(2, 1, 4, dtype=torch.float16),
        prefix + ".mlp.experts.0.qzeros": torch.ones(20, dtype=torch.int32),
    }, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "semantic_start_token_id": 10, "semantic_end_token_id": 12,
        "text_config": {"num_hidden_layers": 1, "num_experts": 2},
        "quantization_config": {"quant_method": "drama3-int4"},
    }))
    req = estimate_host_requirements(tmp_path, include_codec=False)
    assert req.expert_bytes == 32 + 16
    assert req.fixed_gpu_bytes == 16 + 8 + 8 + 64


@pytest.mark.parametrize("arguments", [
    {"ram_available_bytes": -1},
    {"commit_available_bytes": -1},
    {"swap_available_bytes": -1},
    {"buffer_bytes": -1},
    {"ram_available_bytes": True},
    {"expert_storage": "ssd"},
])
def test_invalid_budget_input_is_rejected(arguments):
    values = {"ram_available_bytes": 1000, "buffer_bytes": 0}
    values.update(arguments)
    with pytest.raises(ValueError):
        plan_host_memory(**values)


@pytest.mark.parametrize("arguments", [
    {"staged_experts": 0}, {"codec_max_frames": 0},
    {"lm_head_device": "meta"}, {"embedding_device": "disk"},
])
def test_invalid_estimator_placement_is_rejected(arguments):
    with pytest.raises(ValueError):
        estimate_host_requirements(**arguments)
