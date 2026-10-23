"""Simulate failed graph capture without allocating CUDA tensors."""

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from drama_low_vram.engine import DramaEngine, EngineOptions
from drama_low_vram.tower_graph import TowerDecodeGraph, TowerGraphRecoveryError


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery tests must not access the CUDA device")

    for name in (
        "is_available", "current_device", "device", "synchronize",
        "is_current_stream_capturing", "graph_pool_handle",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def partial_graph():
    graph = object.__new__(TowerDecodeGraph)
    graph.tower = SimpleNamespace(device=torch.device("cuda:0"))
    graph.graphs = {256: object()}
    graph.pool = object()
    graph.logits = object()
    graph.started = True
    graph.closed = False
    return graph


@pytest.mark.parametrize("error", [
    torch.cuda.OutOfMemoryError("partial capture allocation"),
    RuntimeError("capture_end failed"),
])
def test_safe_recovery_preserves_original_error_and_releases_partial_graphs(error):
    graph = partial_graph()
    graph._capture = Mock(side_effect=error)

    def recovery():
        assert graph.graphs == {}
        assert graph.pool is None
        assert graph.logits is None
        assert graph.closed and not graph.started

    graph._check_cuda_capture_recovery = Mock(side_effect=recovery)
    with pytest.raises(type(error)) as raised:
        graph._capture_with_recovery()
    assert raised.value is error
    graph._check_cuda_capture_recovery.assert_called_once_with()


def test_failed_recovery_is_not_an_oom_that_can_silently_select_eager():
    graph = partial_graph()
    capture_error = torch.cuda.OutOfMemoryError("partial capture allocation")
    graph._capture = Mock(side_effect=capture_error)
    graph._check_cuda_capture_recovery = Mock(
        side_effect=RuntimeError("Philox generator remains in capture mode")
    )
    with pytest.raises(TowerGraphRecoveryError, match="Restart the engine process") as raised:
        graph._capture_with_recovery()
    assert not isinstance(raised.value, torch.cuda.OutOfMemoryError)
    assert raised.value.__cause__ is capture_error
    assert "Philox generator" in str(raised.value)
    assert graph.graphs == {} and graph.pool is None and graph.closed


def test_partial_capture_locals_are_released_before_recovery_probe():
    graph = partial_graph()
    captured = []

    class Allocation:
        pass

    def fail_capture():
        local_allocation = Allocation()
        captured.append(weakref.ref(local_allocation))
        raise torch.cuda.OutOfMemoryError("local capture allocation")

    def recovery():
        assert captured[0]() is None

    graph._capture = fail_capture
    graph._check_cuda_capture_recovery = recovery
    with pytest.raises(torch.cuda.OutOfMemoryError):
        graph._capture_with_recovery()


def test_successful_capture_does_not_close_or_probe():
    graph = partial_graph()
    graph._capture = Mock()
    graph._check_cuda_capture_recovery = Mock()
    graph._capture_with_recovery()
    graph._capture.assert_called_once_with()
    graph._check_cuda_capture_recovery.assert_not_called()
    assert graph.graphs and graph.pool is not None and not graph.closed


def test_recovery_checks_idle_stream_execution_and_rng_without_changing_rng(monkeypatch):
    graph = partial_graph()
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    random_tensor = SimpleNamespace(uniform_=lambda: events.append("rng"))
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: random_tensor)
    fork_rng = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    graph._check_cuda_capture_recovery()
    fork_rng.assert_called_once_with(devices=[0])
    assert events == ["sync", "rng", "sync"]


def test_active_capture_stream_fails_before_allocating_or_synchronizing(monkeypatch):
    graph = partial_graph()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="remains in capture mode"):
        graph._check_cuda_capture_recovery()


def test_rng_probe_failure_is_reported(monkeypatch):
    graph = partial_graph()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())

    def broken_rng():
        raise RuntimeError("Philox state invalid")

    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: SimpleNamespace(uniform_=broken_rng)
    )
    with pytest.raises(RuntimeError, match="Philox state invalid"):
        graph._check_cuda_capture_recovery()


def test_cpu_recovery_is_noop():
    graph = partial_graph()
    graph.tower.device = torch.device("cpu")
    graph._check_cuda_capture_recovery()


def test_engine_rejects_reuse_after_tower_recovery_failure(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = SimpleNamespace(
        cache_capacity=64,
        config=SimpleNamespace(text=SimpleNamespace(
            max_position_embeddings=128, vocab_size=100,
        )),
        backbone=SimpleNamespace(lm_head=object()),
        audio_decoder=object(),
    )
    engine.manager = SimpleNamespace(
        release_runtime_buffers=Mock(), configure_staging=Mock(),
    )
    engine._resume_model = Mock()
    engine._batching = Mock(return_value=(4, 32, 16))
    monkeypatch.setattr("drama_low_vram.engine.ensure_drama3_caches", Mock())
    planner = Mock()
    monkeypatch.setattr("drama_low_vram.engine.configure_model_runtime", planner)
    failure = TowerGraphRecoveryError(
        "Tower CUDA/RNG state did not recover. Restart the engine process."
    )
    monkeypatch.setattr(
        "drama_low_vram.tower_graph.TowerDecodeGraph", Mock(side_effect=failure),
    )
    with pytest.raises(TowerGraphRecoveryError) as raised:
        engine._prepare(32)
    assert raised.value is failure
    assert engine.graph is None and engine._ready_plan is None
    assert engine._transition_failure == str(failure)
    planner.assert_not_called()
    with pytest.raises(RuntimeError, match="Restart the engine process"):
        engine.load()
    with pytest.raises(RuntimeError, match="Restart the engine process"):
        engine.prepare_reference(tmp_path / "missing.wav", "Reference speech.")
