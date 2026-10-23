"""Reference boundary identity, LRU residency and memory-pressure fallback."""

import pytest
import torch

from drama_low_vram.prefix_cache import ReferencePrefixCache, prefix_execution_key
from drama_low_vram.sequence_state import SequenceSnapshot
from drama_low_vram.tower_graph import TowerDecodeGraph
from test_sequence_state import model
from drama_low_vram.generation import generate, prefill_reference_prefix
from test_backbone_graph import tiny_config
from test_tower_graph import tiny_tower
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.generate import ensure_drama3_caches
from types import SimpleNamespace


def prompt():
    rows = torch.zeros(10, 5, dtype=torch.int64)
    rows[:, 0] = 7
    rows[6, 0] = 3
    return rows


def test_only_complete_system_prefix_chunks_and_lookahead_are_shared():
    value, rows = model(), prompt()
    cache = ReferencePrefixCache()
    key, boundary = cache.key(value, rows, 2, ("official",))
    assert boundary == 6
    changed = rows.clone()
    changed[7:] += 1
    assert cache.key(value, changed, 2, ("official",))[0] == key
    changed[6, 1] += 1
    assert cache.key(value, changed, 2, ("official",))[0] != key
    assert cache.key(value, rows, 2, ("other",))[0] != key
    assert cache.key(value, rows, 4, ("official",))[1] == 4
    assert cache.key(model(), rows, 2, ("official",))[0] != key
    rows[0, 0] = 3
    assert cache.key(value, rows, 2) == (None, 0)


def test_residency_changes_keep_prefix_but_arithmetic_and_batching_invalidate_it():
    value = model()
    backend = SimpleNamespace(decode_specialization=False)
    value._drama_low_vram_manager = SimpleNamespace(
        backend=backend, max_staged_experts=8, max_pair_rows=64, cache_experts=2)
    original = prefix_execution_key(value)
    manager = value._drama_low_vram_manager
    manager.cache_experts = 1
    assert prefix_execution_key(value) == original
    manager.max_pair_rows = 16
    assert prefix_execution_key(value) != original
    manager.max_pair_rows = 64
    backend.decode_specialization = True
    assert prefix_execution_key(value) != original


def test_lru_budget_includes_snapshot_and_eviction_precedes_allocation():
    value, rows = model(), prompt()
    size = SequenceSnapshot.estimate_bytes(value, 6, 6)
    cache = ReferencePrefixCache(size)
    key, boundary = cache.key(value, rows, 2)
    assert cache.get(key) is None
    assert cache.capture(key, value, boundary)
    assert cache.get(key).nbytes == size and cache.bytes == size
    rows[0, 1] = 1
    second, _ = cache.key(value, rows, 2)
    assert cache.capture(second, value, boundary)
    assert cache.get(key) is None
    assert cache.report()["entries"] == 1 and cache.report()["evictions"] == 1
    assert cache.bytes <= cache.budget_bytes
    cache.clear()
    assert cache.bytes == 0 and not cache.entries


def test_disabled_suspended_and_memory_pressure_skip_without_model_mutation():
    value, rows = model(), prompt()
    assert ReferencePrefixCache(0).key(value, rows, 2) == (None, 0)
    cache = ReferencePrefixCache(admission=lambda _size: False)
    key, boundary = cache.key(value, rows, 2)
    assert not cache.capture(key, value, boundary)
    assert cache.skips == 1 and cache.bytes == 0
    with cache.suspend():
        assert cache.key(value, rows, 2) == (None, 0)
        with cache.suspend():
            assert cache.key(value, rows, 2) == (None, 0)
    assert cache.key(value, rows, 2)[0] == key
    assert not ReferencePrefixCache(1).capture(key, value, boundary)
    with pytest.raises(ValueError, match="non-negative"):
        ReferencePrefixCache(-1)


@pytest.mark.parametrize("device,dtype,use_graph", [
    ("cpu", torch.float32, False),
    pytest.param("cuda", torch.bfloat16, False,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cuda", torch.float16, True,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@torch.inference_mode()
def test_generation_prefix_reuse_changes_only_prefill_work(
        monkeypatch, device, dtype, use_graph):
    torch.manual_seed(42)
    cfg = tiny_config()
    cfg.hidden_size, cfg.max_position_embeddings = 32, 32
    value = torch.nn.Module()
    value.backbone = FishOmniBackbone(
        cfg, fp8_experts=False, dtype=dtype, device=device).eval()
    for layer in value.backbone.model.layers:
        torch.nn.init.normal_(layer.mlp.experts.w13_weight, std=0.04)
        torch.nn.init.normal_(layer.mlp.experts.w2_weight, std=0.04)
    value.audio_decoder = tiny_tower(device).to(dtype=dtype)
    value.config = SimpleNamespace(text=cfg, eos_token_id=3)
    value.cache_capacity, value.decode_graphs = 0, None
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    rows = prompt()
    cache = ReferencePrefixCache()
    if use_graph:
        ensure_drama3_caches(value, 32)
    graph = (TowerDecodeGraph(
        value.audio_decoder, 32, logits_size=cfg.vocab_size) if use_graph else None)

    def logits(hidden):
        output = value.backbone.compute_logits(hidden)
        output[:, value.config.eos_token_id] = torch.finfo(output.dtype).min
        return output

    def run(use):
        torch.manual_seed(123)
        return generate(value, rows, max_new_tokens=4, prefill_chunk_size=2,
                        top_k=1, prefix_cache=cache if use else None,
                        tower_graph=graph, logit_head=logits)

    try:
        warmup = prefill_reference_prefix(
            value, rows, chunk_size=2, prefix_cache=cache)
        baseline = run(False)
        hit = run(True)
        assert len(baseline.frames) > 0 and torch.equal(baseline.frames, hit.frames)
        assert warmup["status"] == "ready" and warmup["cached_rows"] == 6
        assert hit.prefix_cache["hit"] and hit.prefix_cache["reused_rows"] == 6
        key_before = len(cache.entries)
        rows[8, 0] = 8
        changed = run(True)
        assert changed.prefix_cache["hit"] and len(cache.entries) == key_before
        expected = run(False)
        assert torch.equal(changed.frames, expected.frames)
    finally:
        if graph is not None:
            graph.close()
