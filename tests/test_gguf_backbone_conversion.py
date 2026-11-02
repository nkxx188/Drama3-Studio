"""The performance recipe must not accidentally quantize protected tensors."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_gguf_backbone_q8 import (
    bf16_to_float32,
    build_plan,
    projection_group,
    verify_output,
)
from drama_low_vram.gguf_archive import Tensor, pack_string, write_header


@pytest.mark.parametrize("module,group", [
    ("self_attn.q_proj", "self_attn"),
    ("self_attn.k_proj", "self_attn"),
    ("self_attn.v_proj", "self_attn"),
    ("self_attn.o_proj", "self_attn"),
    ("linear_attn.in_proj_qkv", "linear_attn"),
    ("linear_attn.in_proj_z", "linear_attn"),
    ("linear_attn.in_proj_b", "linear_attn"),
    ("linear_attn.in_proj_a", "linear_attn"),
    ("linear_attn.out_proj", "linear_attn"),
    ("mlp.shared_expert.gate_proj", "shared_expert"),
    ("mlp.shared_expert.up_proj", "shared_expert"),
    ("mlp.shared_expert.down_proj", "shared_expert"),
])
def test_projection_selection_is_explicit(module, group):
    name = f"model_weights/model.language_model.layers.12.{module}.weight"
    assert projection_group(name) == group


@pytest.mark.parametrize("name", [
    "model_weights/model.language_model.layers.0.mlp.experts.gate_proj.weight",
    "model_weights/model.language_model.layers.0.mlp.experts.7.up_proj.weight",
    "model_weights/model.language_model.layers.0.mlp.gate.weight",
    "model_weights/model.language_model.layers.0.mlp.shared_expert_gate.weight",
    "model_weights/model.language_model.layers.0.linear_attn.conv1d.weight",
    "model_weights/model.language_model.layers.0.linear_attn.A_log",
    "model_weights/model.language_model.layers.0.input_layernorm.weight",
    "model_weights/model.language_model.embed_tokens.weight",
    "model_weights/lm_head.weight",
    "model_weights/audio_decoder_mtp.heads.0.weight",
    "codec_weights/quantizer.semantic_quantizer.quantizers.0.codebook.weight",
])
def test_protected_tensors_are_not_selected(name):
    assert projection_group(name) is None


def test_plan_changes_only_selected_projection_and_preserves_source_objects():
    name = "model_weights/model.language_model.layers.0.self_attn.q_proj.weight"
    source = [
        Tensor("q", name, (2, 32), 30, 1024, 128),
        Tensor("head", "model_weights/lm_head.weight", (2, 32), 30, 2048, 128),
        Tensor("expert", "model_weights/experts", (2, 32), 12, 3072, 36),
    ]
    plan = build_plan(SimpleNamespace(tensors=source))
    assert [t.kind for t in plan] == [8, 30, 12]
    assert [t.nbytes for t in plan] == [68, 128, 36]
    assert [t.offset for t in plan] == [0, 0, 0]
    assert source[0].kind == 30 and source[0].offset == 1024


@pytest.mark.parametrize("shape,kind", [((2, 31), 30), ((2, 32), 8), ((2, 2, 32), 30)])
def test_plan_refuses_requantization_or_invalid_geometry(shape, kind):
    name = "model_weights/model.language_model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match="block-aligned BF16"):
        build_plan(SimpleNamespace(tensors=[Tensor("q", name, shape, kind, 0, 128)]))


def test_bf16_decode_preserves_sign_and_exponent():
    values = np.array([1.0, -2.0, 0.0, 65536.0], dtype=np.float32)
    raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
    assert np.array_equal(bf16_to_float32(raw, (2, 2)), values.reshape(2, 2))


def test_output_verifier_detects_payload_corruption(tmp_path):
    import hashlib
    import struct

    names = ["model_weights/a"]
    metadata = [("audiocpp.tensor_names", pack_string("audiocpp.tensor_names")
                 + struct.pack("<IIQ", 9, 8, 1) + pack_string(names[0]))]
    plan = [Tensor("physical", names[0], (1, 32), 8, 0, 34)]
    path = tmp_path / "output.gguf"
    payload = b"\x01" * 34
    with path.open("wb") as file:
        size = write_header(file, metadata, plan)
        file.seek(plan[0].offset)
        file.write(payload)
        file.write(bytes(size - file.tell()))
    records = [{"sha256": hashlib.sha256(payload).hexdigest()}]
    assert verify_output(path, plan, records) == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r+b") as file:
        file.seek(plan[0].offset)
        file.write(b"\x02")
    with pytest.raises(ValueError, match="verification failed"):
        verify_output(path, plan, records)
