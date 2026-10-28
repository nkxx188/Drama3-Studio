"""Resource accounting without GPU/model allocations."""

from dataclasses import replace

import pytest

from drama_low_vram.gguf_resources import Layout, MIB, GIB, plan_resources, codec_capability
from drama_low_vram.gguf_archive import Tensor


def layout(q8=True):
    return Layout("model.gguf", 40 * GIB, 40, 256, 32640 * MIB, int(127.5 * MIB),
                  7074 * MIB, 1326 * MIB, 793 * MIB, 954 * MIB, 986 * MIB,
                  q8, (8,) if q8 else (12,), codec_cuda_supported=True, host_embedding_supported=True)


def test_available_memory_is_not_reduced_by_system_usage_again():
    plan = plan_resources(layout(), free_vram=int(8.5 * GIB), available_ram=44 * GIB, cores=8)
    assert plan.host_storage == "pin"
    assert plan.available_ram_bytes == 44 * GIB
    assert plan.host_embedding and plan.decoder_backend == "cuda"
    assert plan.session_options()["fish_drama3.cpu_expert_layers"] == "0"
    assert "fish_drama3.decode_cpu_experts" not in plan.session_options()


def test_budgets_follow_model_size_and_gpu_capacity():
    low = plan_resources(layout(), free_vram=int(7.5 * GIB), available_ram=24 * GIB, cores=2)
    high = plan_resources(layout(), free_vram=24 * GIB, available_ram=48 * GIB, cores=32)
    assert low.host_storage == "mapped" and low.threads == 2
    assert high.cache_experts > low.cache_experts >= 8
    assert high.resident_estimate_bytes + high.workspace_estimate_bytes <= 24 * GIB
    assert high.threads == 8


def test_q4_uses_compatible_codec_and_embedding_independently_of_expert_format():
    q4 = replace(layout(False), dense_bytes=5814 * MIB, expert_bytes=17280 * MIB,
                 slot_bytes=int(67.5 * MIB))
    plan = plan_resources(q4, free_vram=int(7.5 * GIB), available_ram=30 * GIB, cores=8)
    assert plan.host_embedding and plan.decoder_backend == "cuda"
    assert plan.codec_backend == "cpu" and plan.codec_strategy == "stream"
    assert not plan.layout.validated_q8


def test_cuda_codec_keeps_separate_graph_headroom_and_bounds_prefill_work():
    model = replace(layout(False), dense_bytes=5814 * MIB, expert_bytes=17280 * MIB,
                    slot_bytes=int(67.5 * MIB))
    plan = plan_resources(model, free_vram=int(7.0 * GIB), available_ram=40 * GIB, cores=8)
    assert plan.decoder_backend == "cuda"
    assert plan.workspace_estimate_bytes == 1024 * MIB + 1024 * 128 * 1024
    assert plan.prefill_work_width == 8
    assert plan.report()["prefill_work_width"] == 8
    assert plan.cache_experts > plan.prefill_work_width
    assert plan.resident_estimate_bytes + plan.workspace_estimate_bytes <= plan.free_vram_bytes
    high = plan_resources(model, free_vram=24 * GIB, available_ram=48 * GIB, cores=8)
    assert high.prefill_work_width == high.cache_experts


def test_prefill_width_scales_with_actual_post_plan_vram_headroom():
    model = replace(layout(False), dense_bytes=5814 * MIB, expert_bytes=17280 * MIB,
                    slot_bytes=int(67.5 * MIB))
    plans = [
        plan_resources(model, free_vram=int(gib * GIB), available_ram=48 * GIB, cores=8)
        for gib in (7, 9, 10, 11, 12, 14)
    ]
    widths = [plan.prefill_work_width for plan in plans]
    assert widths == sorted(widths)
    assert widths[:2] == [8, 8]
    assert 8 < widths[2] <= 10
    assert widths[2] < plans[2].cache_experts
    assert widths[3] > widths[2]
    assert widths[4] > widths[3]
    assert all(plan.prefill_workspace_bytes ==
               (plan.prefill_work_width - 8) * model.slot_bytes for plan in plans)
    assert all(plan.resident_estimate_bytes + plan.workspace_estimate_bytes
               <= plan.free_vram_bytes for plan in plans)


@pytest.mark.parametrize("kinds", [(2,), (6,), (11,), (12,), (13,), (14,), (12, 14)])
def test_codec_policy_does_not_depend_on_expert_quantization(kinds):
    model = replace(layout(False), expert_types=kinds)
    plan = plan_resources(model, free_vram=12 * GIB, available_ram=24 * GIB, cores=4)
    assert plan.decoder_backend == "cuda"


def test_unsupported_codec_stays_on_cpu_even_for_q8():
    model = replace(layout(), codec_cuda_supported=False, codec_note="Quantized codec")
    plan = plan_resources(model, free_vram=24 * GIB, available_ram=48 * GIB, cores=8)
    assert plan.decoder_backend == "cpu"
    with pytest.raises(ValueError, match="not compatible"):
        plan_resources(model, free_vram=24 * GIB, available_ram=48 * GIB, cores=8, decoder="cuda")


def test_budget_respects_explicit_cache_and_cpu_override():
    model = layout()
    plan = plan_resources(model, free_vram=int(7.5 * GIB), available_ram=24 * GIB, cores=4,
                          cache=12)
    assert plan.decoder_backend == "cpu" and plan.cache_experts == 12
    plan = plan_resources(model, free_vram=24 * GIB, available_ram=48 * GIB, cores=8, decoder="cpu")
    assert plan.decoder_backend == "cpu" and plan.codec_strategy == "full"
    with pytest.raises(MemoryError, match="Requested CUDA codec"):
        plan_resources(model, free_vram=int(7.5 * GIB), available_ram=48 * GIB, cores=8,
                       cache=12, decoder="cuda")


def test_insufficient_codec_budget_keeps_full_top8_on_gpu():
    model = layout()
    minimum = (model.dense_bytes - model.compact_head_saved_bytes - model.host_embedding_saved_bytes
               + 8 * model.slot_bytes + 640 * MIB)
    plan = plan_resources(model, free_vram=minimum, available_ram=16 * GIB, cores=4)
    assert plan.decoder_backend == "cpu" and plan.cache_experts == 8
    assert plan.session_options()["fish_drama3.cpu_expert_layers"] == "0"


def codec_tensors():
    anchors = {
        "encoder.block.0.conv.weight": (64, 1, 7),
        "encoder.block.6.conv.weight": (1024, 1024, 3),
        "decoder.model.0.conv.weight": (1536, 1024, 7),
        "decoder.model.5.alpha": (96, 1),
        "decoder.model.6.conv.weight": (96, 7),
        "quantizer.semantic_quantizer.quantizers.0.codebook.weight": (4096, 8),
        **{f"quantizer.quantizer.quantizers.{i}.codebook.weight": (1024, 8) for i in range(9)},
        **{f"quantizer.{module}.layers.0.attention.wqkv.weight": (3072, 1024)
           for module in ("pre_module", "post_module")},
    }
    return [Tensor(name, "codec_weights/" + name, shape, 1, 0, 0)
            for name, shape in anchors.items()]


def test_codec_capability_handles_supported_float_types_and_rejects_missing_or_quantized_codec():
    tensors = codec_tensors()
    assert codec_capability(tensors)[0]
    assert codec_capability([replace(t, kind=30) for t in tensors])[0]
    assert codec_capability([replace(t, kind=0) for t in tensors])[0]
    assert not codec_capability(tensors[1:])[0]
    assert not codec_capability([replace(t, kind=8) for t in tensors])[0]
    assert not codec_capability([])[0]
    assert not codec_capability([replace(t, shape=(1,)) for t in tensors])[0]


def test_codec_signature_ignores_experts_and_detects_codec_changes():
    tensors = codec_tensors()
    signature = codec_capability(tensors)[1]
    assert codec_capability(tensors + [Tensor("expert", "model_weights/experts", (256,), 12, 0, 144)])[1] == signature
    assert codec_capability([replace(t, kind=0) for t in tensors])[1] != signature


def test_codec_accepts_split_qkv_as_well_as_packed():
    tensors = [t for t in codec_tensors() if "wqkv" not in t.logical]
    tensors += [Tensor("split", f"codec_weights/quantizer.{module}.layers.0.attention.{projection}.weight",
                       (1024, 1024), 1, 0, 0)
                for module in ("pre_module", "post_module") for projection in ("q_proj", "k_proj", "v_proj")]
    assert codec_capability(tensors)[0]
    assert not codec_capability(tensors[:-1])[0]


def test_fallbacks_are_finite_monotonic_and_preserve_top8():
    plan = plan_resources(layout(), free_vram=16 * GIB, available_ram=48 * GIB, cores=8)
    options = [plan, *plan.fallbacks()]
    assert len(options) == 4
    assert options[-1].host_storage == "mapped" and options[-1].decoder_backend == "cpu"
    assert options[-1].cache_experts == 8
    assert options[-2].prefill_work_width == 8
    assert options[-2].prefill_workspace_bytes == 0
    assert all(next.resident_estimate_bytes <= prev.resident_estimate_bytes
               for prev, next in zip(options, options[1:]))
    assert list(options[-1].fallbacks()) == []


def test_impossible_plan_fails_before_allocating():
    with pytest.raises(MemoryError, match="minimum GPU estimate"):
        plan_resources(layout(), free_vram=2 * GIB, available_ram=8 * GIB, cores=8)
    with pytest.raises(MemoryError, match="Requested expert cache"):
        plan_resources(layout(), free_vram=8 * GIB, available_ram=48 * GIB, cores=8, cache=256)


@pytest.mark.parametrize("available,retained", [(24, 256), (12, 64), (4, 0)])
def test_optional_prefill_retention_shrinks_under_ram_pressure(available, retained):
    plan = plan_resources(layout(), free_vram=12 * GIB, available_ram=available * GIB, cores=4)
    assert plan.prefill_cache_mib == retained
    assert plan.session_options()["fish_drama3.backbone_prefill_cache_mb"] == str(retained)
    assert plan.session_options()["fish_drama3.tower_prefill_cache_mb"] == str(retained)
