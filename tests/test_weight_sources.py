"""File-backed immutable weights retain values and mapping lifetime."""

import sys
import torch
import pytest
from safetensors.torch import save_file

from drama_low_vram.checkpoint_reader import iter_readonly_file_tensors
from drama_low_vram.model import _load_dense_backbone
from drama_low_vram.weight_sources import ImmutableWeightSources, reclaimable_file_bytes


def test_cpu_embedding_binds_exact_file_rows_without_materializing_full_table(tmp_path):
    original = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
    projection = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    path = tmp_path / "weights.safetensors"
    save_file({
        "model.language_model.embed_tokens.weight": original,
        "model.language_model.projection.weight": projection,
    }, str(path))
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.embed_tokens = torch.nn.Embedding(12, 4, dtype=torch.bfloat16, device="meta")
    root.model.projection = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    sources = ImmutableWeightSources()
    mapped = dict(iter_readonly_file_tensors(path))
    _load_dense_backbone(
        root, iter(mapped.items()), sources=sources, mapped_embedding=True)
    embedding = root.model.embed_tokens.weight
    assert embedding.device.type == "cpu" and not embedding.requires_grad
    assert embedding.data_ptr() == mapped["model.language_model.embed_tokens.weight"].data_ptr()
    assert torch.equal(root.model.embed_tokens(torch.tensor([0, 5, 11])), original[[0, 5, 11]])
    assert torch.equal(root.model.projection.weight, projection)
    assert sources.get(root.model.projection, "parameter", "weight").data_ptr() == (
        mapped["model.language_model.projection.weight"].data_ptr())
    assert sources.as_dict()["cpu_embedding_bytes"] == original.numel() * 2
    # The bound Parameter owns its mapping independently of iterators/registry.
    del mapped
    sources.clear()
    assert torch.equal(root.model.embed_tokens(torch.tensor([7])), original[[7]])


def test_different_source_dtype_keeps_the_original_cpu_conversion():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.embed_tokens = torch.nn.Embedding(3, 4, dtype=torch.bfloat16, device="meta")
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    sources = ImmutableWeightSources()
    _load_dense_backbone(
        root, [("model.language_model.embed_tokens.weight", original)],
        sources=sources, mapped_embedding=True)
    assert root.model.embed_tokens.weight.device.type == "cpu"
    assert torch.equal(root.model.embed_tokens.weight, original.bfloat16())
    assert sources.nbytes == 0


def test_invalid_source_shape_is_rejected_without_publishing():
    root = torch.nn.Linear(4, 3, bias=False)
    sources = ImmutableWeightSources()
    with pytest.raises(ValueError, match="shape mismatch"):
        sources.remember(root, "weight", torch.zeros(2, 4))
    assert sources.nbytes == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set query")
def test_resident_file_credit_counts_whole_pages_without_counting_aliases_twice(tmp_path):
    path = tmp_path / "resident.safetensors"
    save_file({"weight": torch.zeros(2 * 1024**2, dtype=torch.uint8)}, str(path))
    source = dict(iter_readonly_file_tensors(path))["weight"]
    assert source[::4096].sum().item() == 0  # Explicitly touch each file page.
    credit = reclaimable_file_bytes([source, source], source.numel() * 2)
    assert 1024**2 <= credit <= source.numel()


def test_prefix_credit_requires_real_file_pages_and_sufficient_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from drama_low_vram.engine import DramaEngine, EngineOptions

    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, prefix_cache_mib=1))
    engine.codec = object()
    engine.model = SimpleNamespace(_drama_low_vram_weight_sources=object())
    expert = torch.nn.Module()
    expert.register_buffer("weight", torch.zeros(128, dtype=torch.uint8))
    expert._drama_expert_storage = "mmap"
    engine.manager = SimpleNamespace(modules={"expert": expert})
    engine.host_plan = SimpleNamespace(
        buffer_bytes=10, requirements=SimpleNamespace(parking_bytes=0))
    hardware = SimpleNamespace(ram_available_bytes=10, commit_available_bytes=100)
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware)
    monkeypatch.setattr("drama_low_vram.weight_sources.reclaimable_file_bytes", lambda *_: 49)
    assert not engine._admit_prefix_snapshot(50)
    monkeypatch.setattr("drama_low_vram.weight_sources.reclaimable_file_bytes", lambda *_: 50)
    assert engine._admit_prefix_snapshot(50)
    assert engine._prefix_admission_report["file_page_admissions"] == 1
    hardware.commit_available_bytes = 59
    assert not engine._admit_prefix_snapshot(50)
    hardware.commit_available_bytes = 100
    def unavailable(*_args):
        raise OSError("Working-set query is unavailable")
    monkeypatch.setattr("drama_low_vram.weight_sources.reclaimable_file_bytes", unavailable)
    assert not engine._admit_prefix_snapshot(50)
    expert._drama_expert_storage = "ram"
    assert not engine._admit_prefix_snapshot(50)
