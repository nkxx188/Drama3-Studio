"""Shared packed weights retain immutable views and original module geometry."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from drama_low_vram.shared_checkpoint import (
    SharedFP8Experts, descriptor, iter_packed, virtual_tensors,
)
from drama_low_vram.checkpoint_reader import iter_readonly_file_tensors


def make_shared(tmp_path):
    values = {
        "model.layers.0.mlp.experts.gate_up_weight": torch.zeros(2, 256, 128, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.experts.down_weight": torch.zeros(2, 128, 128, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.experts.gate_up_scale": torch.ones(2, 2, 1),
        "model.layers.0.mlp.experts.down_scale": torch.ones(2, 1, 1),
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight": torch.arange(16.).reshape(4, 4),
        "tower.shared_layers.0.feed_forward.w13.weight": torch.arange(24.).reshape(6, 4),
        "tower.shared_layers.0.embedding.weight": torch.ones(3, 4),
        "audio_head.weight": torch.ones(3, 4),
    }
    file = tmp_path / "packed.safetensors"
    save_file(values, str(file))
    (tmp_path / "shared-model.json").write_text(json.dumps({
        "format": "drama3-packed-shared-v1", "model": str(file),
        "size": file.stat().st_size, "sha256": "a" * 64,
    }), encoding="utf-8")
    config = SimpleNamespace(text=SimpleNamespace(
        num_hidden_layers=1, num_experts=2, moe_intermediate_size=128, hidden_size=128))
    return file, values, config


def test_virtual_weights_split_fused_rows_without_reordering(tmp_path):
    _, values, _ = make_shared(tmp_path)
    mapped = dict(virtual_tensors(tmp_path))
    assert "audio_head.weight" not in mapped
    assert torch.equal(
        mapped["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"],
        values["model.layers.0.mlp.shared_expert.gate_up_proj.weight"][:2])
    assert torch.equal(
        mapped["audio_decoder_mtp_shared.layers.0.transformer_layer.feed_forward.w3.weight"],
        values["tower.shared_layers.0.feed_forward.w13.weight"][3:])
    assert "audio_decoder_mtp_shared.layers.0.semantic_embedding.weight" in mapped


def test_shared_expert_bank_owns_views_after_iterator_and_owner_close(tmp_path):
    file, _, config = make_shared(tmp_path)
    experts = SharedFP8Experts(tmp_path, config)
    from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
    module = MoEExperts(2, 128, 128, fp8=True, device="meta")
    class Backbone:
        def get_submodule(self, name):
            assert name == "model.layers.0.mlp.experts"
            return module
    assert experts.assign_to_backbone(Backbone()) == 1
    assert module.w13_weight.device.type == "cpu" and module.w13_weight.is_contiguous()
    experts.close()
    assert torch.equal(module.w13_scale_inv, torch.ones(2, 2, 1))
    with file.open("rb") as handle:
        assert len(handle.read(8)) == 8


def test_descriptor_and_reader_reject_truncation_before_exposing_tensors(tmp_path):
    file, _, _ = make_shared(tmp_path)
    with file.open("r+b") as handle:
        handle.truncate(file.stat().st_size - 8)
    with pytest.raises(ValueError, match="size changed"):
        descriptor(tmp_path)
    with pytest.raises(ValueError):
        next(iter_readonly_file_tensors(file))


def test_packed_filter_keeps_only_requested_buffer(tmp_path):
    make_shared(tmp_path)
    assert [name for name, _ in iter_packed(tmp_path, include_prefixes=("audio_head.",))] == [
        "audio_head.weight"]
