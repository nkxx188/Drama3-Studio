"""Actual GDN/KV state and sampling survive a failed automatic prefill."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from drama_low_vram.engine import DramaEngine, EngineOptions
from drama_low_vram.generation import generate, prefill_reference_prefix
from drama_low_vram.prefix_cache import ReferencePrefixCache
from drama_low_vram.sequence_state import SequenceSnapshot
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from test_autotune import post_load_plan
from test_backbone_graph import tiny_config
from test_prefix_cache import prompt
from test_tower_graph import tiny_tower


@pytest.fixture
def small_model(monkeypatch):
    torch.manual_seed(42)
    cfg = tiny_config()
    cfg.hidden_size, cfg.max_position_embeddings = 32, 32
    value = torch.nn.Module()
    value.backbone = FishOmniBackbone(cfg, fp8_experts=False, dtype=torch.float32).eval()
    with torch.no_grad():
        for layer in value.backbone.model.layers:
            torch.nn.init.normal_(layer.mlp.experts.w13_weight, std=0.04)
            torch.nn.init.normal_(layer.mlp.experts.w2_weight, std=0.04)
    value.audio_decoder = tiny_tower()
    value.config = SimpleNamespace(text=cfg, eos_token_id=3)
    value.cache_capacity, value.decode_graphs = 0, None
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    return value


@torch.inference_mode()
def test_reference_warmup_recalculates_boundary_after_oom_without_sampling(
        small_model, monkeypatch):
    value, rows, cache = small_model, prompt(), ReferencePrefixCache()
    original = value.backbone.forward
    positions_seen = []

    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        positions = kwargs["input_pos"]
        positions_seen.append(positions.tolist())
        if len(positions) == 4:
            raise torch.cuda.OutOfMemoryError("warm-up prefill pressure")
        return output

    monkeypatch.setattr(value.backbone, "forward", forward)
    decode = Mock(side_effect=AssertionError("Warm-up must not sample speech"))
    monkeypatch.setattr(value.audio_decoder, "decode_step", decode)
    warmed = prefill_reference_prefix(
        value, rows, chunk_size=4, prefix_cache=cache, allow_oom_fallback=True)
    assert warmed["status"] == "ready"
    assert warmed["chunk_size"] == 2 and warmed["oom_retries"] == 1
    assert warmed["cached_rows"] == 6 and warmed["boundary"] == 6
    assert positions_seen == [[0, 1, 2, 3], [0, 1], [2, 3], [4, 5]]
    decode.assert_not_called()
    positions_seen.clear()
    repeated = prefill_reference_prefix(value, rows, chunk_size=2, prefix_cache=cache)
    assert repeated["hit"] and repeated["status"] == "ready"
    assert not positions_seen


@pytest.mark.parametrize("blocked", ["memory", "cancelled"])
@torch.inference_mode()
def test_reference_warmup_skips_unusable_work(small_model, monkeypatch, blocked):
    cache = ReferencePrefixCache(admission=lambda _size: blocked != "memory")
    forward = Mock(side_effect=AssertionError("No speculative work should run"))
    monkeypatch.setattr(small_model.backbone, "forward", forward)
    result = prefill_reference_prefix(
        small_model, prompt(), chunk_size=2, prefix_cache=cache,
        cancelled=lambda: blocked == "cancelled")
    assert result["status"] == ("cancelled" if blocked == "cancelled" else "skipped_memory")
    forward.assert_not_called()
    assert not cache.entries


@pytest.mark.parametrize("use_prefix", [False, True])
@torch.inference_mode()
def test_retry_restarts_partial_gdn_and_kv_and_preserves_next_sample(
        small_model, monkeypatch, use_prefix):
    value, rows = small_model, prompt()
    cache = ReferencePrefixCache() if use_prefix else None

    def run(chunk, **kwargs):
        torch.manual_seed(123)
        return generate(
            value, rows, max_new_tokens=4, prefill_chunk_size=chunk,
            prefix_cache=cache, **kwargs)

    expected = run(2)
    valid_rows = len(rows) + len(expected.frames)
    expected_state = SequenceSnapshot.capture(
        value, backbone_rows=valid_rows, tower_rows=valid_rows - 1)
    expected_rng = torch.get_rng_state().clone()
    original = value.backbone.forward
    failed = []

    def fail_after_mutating_partial_state(*args, **kwargs):
        output = original(*args, **kwargs)
        positions = kwargs["input_pos"]
        if len(positions) == 4 and int(positions[0]) == 4:
            failed.append(True)
            raise torch.cuda.OutOfMemoryError("injected after GDN and KV writes")
        return output

    monkeypatch.setattr(value.backbone, "forward", fail_after_mutating_partial_state)
    actual = run(4, allow_prefill_fallback=True)
    actual_state = SequenceSnapshot.capture(
        value, backbone_rows=valid_rows, tower_rows=valid_rows - 1)
    assert failed == [True]
    assert actual.prefix_cache["chunk_size"] == 2
    assert actual.prefix_cache["oom_retries"] == 1
    assert actual.prefix_cache["hit"] == use_prefix
    assert torch.equal(actual.frames, expected.frames)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual_state.buffers.keys() == expected_state.buffers.keys()
    assert all(torch.equal(actual_state.buffers[key], expected_state.buffers[key])
               for key in expected_state.buffers)


@pytest.mark.parametrize("error", [
    torch.cuda.OutOfMemoryError("manual chunk"), RuntimeError("CUDA device-side assert"),
])
def test_manual_limits_and_nonallocation_cuda_errors_propagate(small_model, monkeypatch, error):
    forward = Mock(side_effect=error)
    monkeypatch.setattr(small_model.backbone, "forward", forward)
    with pytest.raises(type(error), match=str(error)):
        generate(
            small_model, prompt(), max_new_tokens=4, prefill_chunk_size=4,
            allow_prefill_fallback=not isinstance(error, torch.cuda.OutOfMemoryError))
    assert forward.call_count == 1


def test_auto_prefill_retry_is_bounded_at_one_row(small_model, monkeypatch):
    forward = Mock(side_effect=torch.cuda.OutOfMemoryError("no remaining VRAM"))
    monkeypatch.setattr(small_model.backbone, "forward", forward)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        generate(
            small_model, prompt(), max_new_tokens=4, prefill_chunk_size=4,
            allow_prefill_fallback=True)
    assert forward.call_count == 3


def test_engine_retains_successful_auto_chunk_and_keeps_manual_preference(tmp_path):
    plan = post_load_plan(prefill_chunk=32)
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.runtime = SimpleNamespace(
        max_staged_experts=8, max_pair_rows=64, prefill_chunk=32)
    engine.model, engine.manager = SimpleNamespace(), SimpleNamespace()
    engine._ready_plan = plan
    engine._remember_prefill_chunk({"chunk_size": 16, "oom_retries": 1})
    assert engine._ready_plan.prefill_chunk == 16
    assert engine._batching() == (8, 64, 16)
    assert engine.model._drama_low_vram_runtime_plan is engine._ready_plan
    assert engine.manager.runtime_plan is engine._ready_plan
    # Retained even if a later codec transition drops the runtime plan.
    engine._ready_plan = None
    assert engine._batching()[2] == 16
    engine.options = replace(engine.options, prefill_chunk=24)
    engine._remember_prefill_chunk({"chunk_size": 8, "oom_retries": 1})
    assert engine._batching()[2] == 24
