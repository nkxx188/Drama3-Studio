"""Validate published compressed-bank metadata and codebook staging layout."""

import json
from types import SimpleNamespace

import pytest
import torch

from drama_low_vram.quantized_model import _comfy_store


def bank_values(*, rotation=256):
    values = {}
    table = torch.linspace(-10, 11, 16)
    for name, rows in (("gate_up_proj", 1024), ("down_proj", 512)):
        base = "expert." + name
        values[base + ".weight"] = torch.zeros(2, rows, 256, dtype=torch.int8)
        values[base + ".weight_s_rel"] = torch.ones(2, rows, 32, dtype=torch.uint8)
        values[base + ".weight_s_channel"] = torch.ones(2, rows)
        values[base + ".weight_codebook"] = table.clone()
        configuration = {"format": "asym_w4a8_int8", "group_size": 16,
                         "convrot_groupsize": rotation, "num_experts": 2}
        values[base + ".comfy_quant"] = torch.tensor(
            list(json.dumps(configuration).encode()), dtype=torch.uint8)
    return values, table


def test_w4a8_preserves_codebook_in_each_experts_staged_auxiliary_rows():
    values, table = bank_values()
    text = SimpleNamespace(hidden_size=512, moe_intermediate_size=512, num_experts=2)
    store = _comfy_store(values, "expert", text, 3)
    store.validate()
    assert not values
    for aux in (store.w13_scale_inv, store.w2_scale_inv):
        restored = aux[:, :16, 36:40].contiguous().view(torch.float32).squeeze(-1)
        torch.testing.assert_close(restored, table.expand(2, 16), rtol=0, atol=0)
        assert torch.equal(aux[..., 32:36].contiguous().view(torch.float32),
                           torch.ones(*aux.shape[:2], 1))


def test_w4a8_rejects_a_different_rotation_instead_of_silently_using_h256():
    values, _ = bank_values(rotation=64)
    text = SimpleNamespace(hidden_size=512, moe_intermediate_size=512, num_experts=2)
    with pytest.raises(ValueError, match="ConvRot group"):
        _comfy_store(values, "expert", text, 3)
