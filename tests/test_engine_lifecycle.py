"""Failure rollback uses simulated placements; no CUDA API is called."""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from drama_low_vram.engine import CodecCoexistenceUnavailable, DramaEngine, EngineOptions


class SimulatedTensor:
    def __init__(self, value, device="cpu", *, fail_to=False, fail_cpu=False):
        self.value = value
        self.device = torch.device(device)
        self.fail_to = fail_to
        self.fail_cpu = fail_cpu
        self.transfers = []

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def detach(self):
        return self

    def to(self, device):
        self.transfers.append(("to", str(device)))
        if self.fail_to:
            raise RuntimeError("Simulated H2D failure")
        return SimulatedTensor(self.value, device)

    def cpu(self):
        self.transfers.append(("cpu", "cpu"))
        if self.fail_cpu:
            raise RuntimeError("Simulated D2H failure")
        return SimulatedTensor(self.value, "cpu")


class SimulatedModule:
    def __init__(self, parameters=None, buffers=None):
        self._parameters = parameters or {}
        self._buffers = buffers or {}


@pytest.fixture(autouse=True)
def prevent_cuda_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Lifecycle tests must not access the CUDA device")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "get_device_name",
        "get_device_capability",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
        "manual_seed_all",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    # The simulation substitutes tensor metadata, rather than allocating CUDA.
    monkeypatch.setattr(torch.nn, "Parameter", lambda tensor, requires_grad=False: tensor)


def simulated_engine(tmp_path, *, compact_head=False):
    engine = DramaEngine(
        EngineOptions(checkpoint=tmp_path, compact_head=compact_head)
    )
    embedding = SimulatedModule({"weight": SimulatedTensor("embedding")})
    head = SimulatedModule({"weight": SimulatedTensor("head")})
    expert = SimulatedModule(
        buffers={"weight": SimulatedTensor("immutable-mapped-expert")}
    )
    first = SimulatedModule({"weight": SimulatedTensor("dense-first")})
    second = SimulatedModule(
        {"weight": SimulatedTensor("dense-second")},
        {"buffer": SimulatedTensor("dense-buffer")},
    )
    backbone = SimulatedModule()
    backbone.model = SimpleNamespace(
        embed_tokens=embedding,
        layers=[],
    )
    backbone.lm_head = head
    backbone.inverse_frequencies = {"old-cache": object()}
    tower = SimulatedModule()
    tower.shared_layers = []
    tower.residual_layers = []
    tower.main_candidates = object()
    model = SimulatedModule()
    modules = [model, backbone, embedding, head, expert, first, second, tower]
    model.modules = lambda: iter(modules)
    model.backbone, model.audio_decoder = backbone, tower
    model.cache_capacity = 9
    engine.model = model
    engine.manager = SimpleNamespace(
        modules={"experts": expert},
        configure_hot_cache=Mock(),
        release_runtime_buffers=Mock(),
        pool=SimpleNamespace(close=Mock()),
    )
    return SimpleNamespace(
        engine=engine, model=model, modules=modules, embedding=embedding, head=head,
        expert=expert, first=first, second=second, tower=tower,
    )


def test_failed_resume_rolls_back_partial_gpu_placement_and_can_retry(tmp_path):
    h = simulated_engine(tmp_path)
    engine = h.engine
    engine._parked = True
    original_first = h.first._parameters["weight"]
    original_second = h.second._parameters["weight"]
    engine._cpu_mirrors[(id(h.first), "parameter", "weight")] = original_first
    original_second.fail_to = True
    with pytest.raises(RuntimeError, match="Simulated H2D failure"):
        engine._resume_model()
    assert engine._parked
    assert engine._transition_failure is None
    assert h.first._parameters["weight"] is original_first
    assert h.second._parameters["weight"] is original_second
    assert all(
        not tensor.is_cuda
        for module in h.modules
        for tensor in (*module._parameters.values(), *module._buffers.values())
    )
    assert h.expert._buffers["weight"].transfers == []
    assert h.embedding._parameters["weight"].transfers == []
    assert h.head._parameters["weight"].transfers == []
    assert h.model.cache_capacity == 0
    assert engine.load() is engine

    original_second.fail_to = False
    engine._resume_model()
    assert not engine._parked
    assert h.first._parameters["weight"].is_cuda
    assert h.second._parameters["weight"].is_cuda
    assert h.second._buffers["buffer"].is_cuda
    assert not h.expert._buffers["weight"].is_cuda


def test_failed_parking_poison_guard_blocks_load_and_reference_reuse(tmp_path):
    h = simulated_engine(tmp_path)
    h.first._parameters["weight"] = SimulatedTensor("dense-first", "cuda:0")
    h.second._parameters["weight"] = SimulatedTensor(
        "dense-second", "cuda:0", fail_cpu=True
    )
    with pytest.raises(RuntimeError, match="Simulated D2H failure"):
        h.engine._park_model()
    assert not h.engine._parked
    assert "parking failed" in h.engine._transition_failure
    assert not h.first._parameters["weight"].is_cuda
    assert h.second._parameters["weight"].is_cuda
    with pytest.raises(RuntimeError, match="close this engine"):
        h.engine.load()
    # Failure is rejected before reading a reference path or hitting a cache.
    with pytest.raises(RuntimeError, match="close this engine"):
        h.engine.prepare_reference(Path("missing.wav"), "Reference speech.")


def test_parking_uses_immutable_source_but_copies_unregistered_runtime_buffer(tmp_path):
    h = simulated_engine(tmp_path)
    dense = SimulatedTensor("dense-first", "cuda:0", fail_cpu=True)
    h.first._parameters["weight"] = dense
    runtime = SimulatedTensor("runtime-state", "cuda:0")
    h.second._buffers["buffer"] = runtime
    mapped = SimulatedTensor("dense-first")
    h.model._drama_low_vram_weight_sources = SimpleNamespace(
        get=lambda module, kind, name: mapped
        if module is h.first and kind == "parameter" and name == "weight" else None)
    h.engine._park_model()
    assert h.first._parameters["weight"] is mapped
    assert dense.transfers == []
    assert runtime.transfers == [("cpu", "cpu")]
    assert h.second._buffers["buffer"].value == "runtime-state"
    h.engine._resume_model()
    assert h.first._parameters["weight"].is_cuda
    assert h.second._buffers["buffer"].is_cuda
    h.engine._park_model()
    assert h.first._parameters["weight"] is mapped


class SimulatedCodec:
    def __init__(self, *, transfer_fails=False):
        self.module = SimulatedModule(
            {"weight": SimulatedTensor("codec-weight")},
            {"positions": SimulatedTensor("codec-positions")},
        )
        self._drama_low_vram_max_frames = 128
        self.transfer_fails = transfer_fails

    def modules(self):
        yield self.module

    def to(self, device):
        self.module._parameters["weight"] = self.module._parameters["weight"].to(device)
        if self.transfer_fails:
            raise RuntimeError("Simulated codec staging failure")
        self.module._buffers["positions"] = self.module._buffers["positions"].to(device)
        return self


@pytest.mark.parametrize("staging_failure", [False, True])
def test_codec_failure_restores_original_cpu_tensor_storage(tmp_path, staging_failure):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    codec = SimulatedCodec(transfer_fails=staging_failure)
    engine.codec = codec
    original_weight = codec.module._parameters["weight"]
    original_buffer = codec.module._buffers["positions"]
    message = "staging failure" if staging_failure else "decode failure"
    with pytest.raises(RuntimeError, match=message):
        with engine._codec_on_gpu(16) as staged:
            assert staged.module._parameters["weight"].is_cuda
            assert staged.module._buffers["positions"].is_cuda
            raise RuntimeError("Simulated codec decode failure")
    assert codec.module._parameters["weight"] is original_weight
    assert codec.module._buffers["positions"] is original_buffer
    assert not original_weight.is_cuda
    assert not original_buffer.is_cuda
    assert torch.cuda.synchronize.call_count == 1


@pytest.mark.parametrize("limit", [True, False, 3.5, "3", 0, -1, 1025])
def test_invalid_generation_frame_limit_is_rejected_before_loading(tmp_path, limit):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, codec_max_frames=1024))
    engine.load = Mock()
    with pytest.raises(ValueError):
        engine.synthesize(
            "Valid speech text.", tmp_path / "invalid.wav", max_new_tokens=limit
        )
    assert not engine.load.called


def test_prepare_reuses_ready_plan_without_releasing_experts_or_graphs(tmp_path):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = SimpleNamespace(cache_capacity=1024)
    engine._resume_model = Mock()
    engine._release_graphs = Mock()
    engine.manager = SimpleNamespace(release_runtime_buffers=Mock())
    plan = object()
    engine._ready_plan = plan
    assert engine._prepare(768) == (plan, None)
    assert engine._lifecycle["runtime_reuses"] == 1
    engine.manager.release_runtime_buffers.assert_not_called()
    engine._release_graphs.assert_not_called()


def context_engine(tmp_path, monkeypatch, *, quantum=512, maximum=2048,
                   compute_policy="manual"):
    engine = DramaEngine(EngineOptions(
        checkpoint=tmp_path, tower_graph=False, staged_experts=8,
        pair_rows=64, prefill_chunk=32, context_bucket_size=quantum,
        compute_policy=compute_policy))
    engine.model = SimpleNamespace(
        cache_capacity=0, config=SimpleNamespace(
            text=SimpleNamespace(max_position_embeddings=maximum)))
    engine._resume_model = Mock()
    engine._release_graphs = Mock()
    engine.manager = SimpleNamespace(
        release_runtime_buffers=Mock(), configure_staging=Mock())
    plan = object()
    monkeypatch.setattr("drama_low_vram.engine.configure_model_runtime",
                        Mock(return_value=plan))

    def ensure(model, capacity):
        model.cache_capacity = capacity

    ensure_mock = Mock(side_effect=ensure)
    monkeypatch.setattr("drama_low_vram.engine.ensure_drama3_caches", ensure_mock)
    return engine, ensure_mock, plan


def test_context_bucket_reuses_graph_resources_as_stream_history_grows(tmp_path, monkeypatch):
    engine, ensure, plan = context_engine(tmp_path, monkeypatch)
    assert engine._prepare(227) == (plan, None)
    assert engine.model.cache_capacity == 512
    assert engine._prepare(271) == (plan, None)
    ensure.assert_called_once_with(engine.model, 512)
    engine.manager.release_runtime_buffers.assert_called_once()
    engine._release_graphs.assert_called_once()
    assert engine._lifecycle["runtime_reuses"] == 1


def test_auto_compute_is_once_per_engine_and_nested_preparation_does_not_recurse(
        tmp_path, monkeypatch):
    engine, ensure, plan = context_engine(tmp_path, monkeypatch, compute_policy="auto")

    def calibrate(prepared):
        assert prepared._prepare(271) == (plan, None)
        return {"mode": "auto", "status": "calibrated"}

    policy = Mock(side_effect=calibrate)
    monkeypatch.setattr("drama_low_vram.calibration.ensure_compute_policy", policy)
    assert engine._prepare(227) == (plan, None)
    assert engine._prepare(300) == (plan, None)
    policy.assert_called_once_with(engine)
    ensure.assert_called_once()
    assert engine._compute_initialized and not engine._computing_policy
    assert engine.compute_report["status"] == "calibrated"


def test_quantized_auto_compute_keeps_gpu_math_without_fp8_cpu_calibration(tmp_path, monkeypatch):
    engine, _, plan = context_engine(tmp_path, monkeypatch, compute_policy="auto")
    engine.manager.backend = SimpleNamespace(quantized_mode="W4A16-GPTQ")
    policy = Mock(side_effect=AssertionError("FP8 CPU calibration cannot consume GPTQ weights"))
    monkeypatch.setattr("drama_low_vram.calibration.ensure_compute_policy", policy)
    assert engine._prepare(227) == (plan, None)
    assert engine._prepare(300) == (plan, None)
    policy.assert_not_called()
    assert engine.compute_report["status"] == "quantized_gpu"
    assert engine.compute_report["cpu_experts"] == 0


@pytest.mark.parametrize("quantum,maximum,requested,expected", [
    (0, 2048, 271, 271), (512, 1000, 999, 1000), (512, 2048, 1024, 1024),
])
def test_context_buckets_obey_exact_mode_and_checkpoint_limit(
        tmp_path, monkeypatch, quantum, maximum, requested, expected):
    engine, ensure, _ = context_engine(
        tmp_path, monkeypatch, quantum=quantum, maximum=maximum)
    engine._prepare(requested)
    ensure.assert_called_once_with(engine.model, expected)


def test_bucket_allocation_oom_retries_exact_capacity_after_releasing_partial_caches(
        tmp_path, monkeypatch):
    engine, ensure, _ = context_engine(tmp_path, monkeypatch)
    attempts = []

    def allocate(model, capacity):
        attempts.append(capacity)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("Simulated bucket allocation failure")
        engine._clear_sequence_caches.assert_called_once()
        model.cache_capacity = capacity

    ensure.side_effect = allocate
    engine._clear_sequence_caches = Mock()
    engine._prepare(271)
    assert attempts == [512, 271]
    assert engine.model.cache_capacity == 271
    assert engine._last_prepare_timing["context_bucket_fallback"] is True


@pytest.mark.parametrize("direction", ["all", "decode_latents", "decode_waveform"])
def test_codec_operation_oom_retries_once_after_restoring_and_parking(tmp_path, direction):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = object()
    events = []

    @contextmanager
    def staged(_frames, *, operation="all"):
        assert operation == direction
        events.append("stage")
        try:
            yield object()
        finally:
            events.append("restore")

    def park():
        events.append("park")
        engine._parked = True

    attempts = 0

    def operation(_codec):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.cuda.OutOfMemoryError("activation pressure")
        return "waveform"

    engine._codec_on_gpu = staged
    engine._park_model = park
    assert engine._run_codec(300, operation, direction=direction) == "waveform"
    assert events == ["stage", "restore", "park", "stage", "restore"]
    assert engine._codec_force_park
    assert engine._lifecycle["codec_oom_retries"] == 1


def test_codec_oom_after_parking_is_not_retried_forever(tmp_path):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model, engine._parked = object(), True

    @contextmanager
    def staged(_frames):
        yield object()

    engine._codec_on_gpu = staged
    operation = Mock(side_effect=torch.cuda.OutOfMemoryError("still too large"))
    with pytest.raises(torch.cuda.OutOfMemoryError, match="still too large"):
        engine._run_codec(900, operation)
    assert operation.call_count == 1


def test_live_sequence_guard_refuses_parking_before_any_state_mutation(tmp_path):
    h = simulated_engine(tmp_path)
    engine = h.engine
    engine.graph, engine.backbone_graph = object(), object()
    plan = engine._ready_plan = object()
    graph, backbone_graph = engine.graph, engine.backbone_graph
    with pytest.raises(CodecCoexistenceUnavailable, match="no snapshot"):
        with engine._codec_preserving_sequence():
            engine._park_model()
    assert engine._codec_preserve_depth == 0
    assert not engine._parked
    assert engine.model.cache_capacity == 9
    assert engine._ready_plan is plan
    assert engine.graph is graph and engine.backbone_graph is backbone_graph
    engine.manager.release_runtime_buffers.assert_not_called()


def test_live_codec_oom_restores_ownership_without_parking_or_retry(tmp_path):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model, engine.codec = object(), SimulatedCodec()
    engine._codec_can_coexist = Mock(return_value=True)
    engine._park_model = Mock()
    original = engine.codec.module._parameters["weight"]
    operation = Mock(side_effect=torch.cuda.OutOfMemoryError("Codec activation pressure"))
    with pytest.raises(CodecCoexistenceUnavailable, match="exceeded current VRAM"):
        with engine._codec_preserving_sequence():
            engine._run_codec(16, operation)
    assert operation.call_count == 1
    assert engine.codec.module._parameters["weight"] is original
    assert not engine._codec_force_park and engine._codec_preserve_depth == 0
    assert engine._lifecycle["codec_oom_retries"] == 0
    engine._park_model.assert_not_called()


def test_live_codec_can_reclaim_experts_but_cannot_park_when_still_too_large(tmp_path):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model, engine.codec = object(), SimulatedCodec()
    engine.manager = SimpleNamespace(
        stats=SimpleNamespace(hot_cache_bytes=64, device_buffer_bytes=16),
        release_runtime_buffers=Mock())
    engine._codec_can_coexist = Mock(side_effect=[False, True, False])
    engine._park_model = Mock()
    original = engine.codec.module._parameters["weight"]
    with pytest.raises(CodecCoexistenceUnavailable, match="snapshot is required"):
        with engine._codec_preserving_sequence():
            engine._run_codec(16, Mock())
    engine.manager.release_runtime_buffers.assert_called_once()
    engine._park_model.assert_not_called()
    assert engine.codec.module._parameters["weight"] is original
    assert engine._lifecycle["codec_cache_reclaims"] == 1
    assert engine._lifecycle["codec_park_calls"] == 0


def test_nested_live_guards_restore_the_outer_guard_after_failure(tmp_path):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = object()
    with engine._codec_preserving_sequence():
        with pytest.raises(ValueError):
            with engine._codec_preserving_sequence():
                raise ValueError("Inner caller failed")
        assert engine._codec_preserve_depth == 1
        with pytest.raises(CodecCoexistenceUnavailable):
            engine._park_model()
    assert engine._codec_preserve_depth == 0


def test_completed_codec_reclaims_sequence_before_gpu_model_weights(tmp_path):
    h = simulated_engine(tmp_path)
    h.model._drama_low_vram_weight_sources = object()
    h.first._parameters["weight"] = SimulatedTensor("keep-on-gpu", "cuda:0", fail_cpu=True)
    h.engine.codec = SimulatedCodec()
    h.engine.manager.stats = SimpleNamespace(hot_cache_bytes=0, device_buffer_bytes=0)
    h.engine._codec_completed_depth = 1
    h.engine._codec_can_coexist = lambda *_args, **_kwargs: h.model.cache_capacity == 0
    h.engine._park_model = Mock()
    with h.engine._codec_on_gpu(16):
        assert h.model.cache_capacity == 0
        assert h.first._parameters["weight"].is_cuda
    h.engine._park_model.assert_not_called()
    assert h.engine._lifecycle["codec_sequence_reclaims"] == 1
    assert h.engine._lifecycle["codec_park_calls"] == 0


def test_completed_codec_does_not_reclaim_a_guarded_live_sequence(tmp_path):
    h = simulated_engine(tmp_path)
    h.model._drama_low_vram_weight_sources = object()
    h.engine.codec = SimulatedCodec()
    h.engine.manager.stats = SimpleNamespace(hot_cache_bytes=0, device_buffer_bytes=0)
    h.engine._codec_completed_depth = 1
    h.engine._codec_can_coexist = lambda *_args, **_kwargs: False
    with h.engine._codec_preserving_sequence():
        with pytest.raises(CodecCoexistenceUnavailable):
            with h.engine._codec_on_gpu(16):
                raise AssertionError("Admission should refuse before staging")
    assert h.model.cache_capacity == 9
    assert h.engine._lifecycle["codec_sequence_reclaims"] == 0


def test_failed_completed_sequence_reclamation_blocks_future_reuse(tmp_path):
    h = simulated_engine(tmp_path)
    h.model._drama_low_vram_weight_sources = object()
    h.engine.codec = SimulatedCodec()
    h.engine.manager.stats = SimpleNamespace(hot_cache_bytes=0, device_buffer_bytes=0)
    h.engine._codec_completed_depth = 1
    h.engine._codec_can_coexist = lambda *_args, **_kwargs: False
    h.engine._clear_sequence_caches = Mock(side_effect=RuntimeError("cache release failed"))
    with pytest.raises(RuntimeError, match="cache release failed"):
        with h.engine._codec_on_gpu(16):
            raise AssertionError("Staging must not run after reclamation failed")
    assert "reclamation failed" in h.engine._transition_failure
    with pytest.raises(RuntimeError, match="close this engine"):
        h.engine.load()


def test_codec_coexist_uses_current_free_and_reusable_allocator_memory(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = object()
    engine.codec = torch.nn.Linear(4, 4)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (100 * 1024**2, 8 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 400 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 200 * 1024**2)
    assert engine._codec_can_coexist(16)
    assert not engine._codec_can_coexist(128)
    engine._codec_force_park = True
    assert not engine._codec_can_coexist(16)


@pytest.mark.parametrize(
    "admissions,reclaims,parks,probe_release_counts",
    [
        pytest.param([True], 0, 0, [0], id="direct-coexist"),
        pytest.param([False, True, True], 1, 0, [0, 0, 1], id="reclaim-then-coexist"),
        pytest.param([False, True, False], 1, 1, [0, 0, 1], id="reclaim-then-park"),
        pytest.param([False, False], 0, 1, [0, 0], id="reclaim-cannot-fit"),
    ],
)
def test_codec_reclaims_expert_buffers_before_parking(
    tmp_path, admissions, reclaims, parks, probe_release_counts
):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = object()
    engine.codec = codec = SimulatedCodec()
    engine.manager = SimpleNamespace(
        stats=SimpleNamespace(
            hot_cache_bytes=64 * 1024**2, device_buffer_bytes=16 * 1024**2
        ),
        release_runtime_buffers=Mock(),
    )
    plan = engine._ready_plan = object()
    tower_graph = engine.graph = object()
    backbone_graph = engine.backbone_graph = object()
    engine._release_graphs = Mock()
    engine._park_model = Mock()
    original_weight = codec.module._parameters["weight"]
    original_buffer = codec.module._buffers["positions"]
    remaining_admissions = iter(admissions)
    observed_release_counts = []

    def can_coexist(*_args, **_kwargs):
        observed_release_counts.append(engine.manager.release_runtime_buffers.call_count)
        return next(remaining_admissions)

    engine._codec_can_coexist = Mock(side_effect=can_coexist)
    with engine._codec_on_gpu(16) as staged:
        assert staged is codec
        assert staged.module._parameters["weight"].is_cuda
        assert staged.module._buffers["positions"].is_cuda
        assert engine.manager.release_runtime_buffers.call_count == reclaims
        assert engine._park_model.call_count == parks

    calls = [((16,), {})]
    if not admissions[0]:
        calls.append(((16,), {"reclaimable_bytes": 80 * 1024**2}))
    if reclaims:
        calls.append(((16,), {}))
    assert [(call.args, call.kwargs) for call in engine._codec_can_coexist.call_args_list] == calls
    assert observed_release_counts == probe_release_counts
    assert engine._lifecycle["codec_cache_reclaims"] == reclaims
    assert engine._lifecycle["codec_park_calls"] == parks
    assert engine._lifecycle["codec_coexist_calls"] == 1 - parks
    if reclaims:
        assert engine._ready_plan is None
    elif not parks:
        assert engine._ready_plan is plan
    if not parks:
        assert engine.graph is tower_graph
        assert engine.backbone_graph is backbone_graph
        engine._release_graphs.assert_not_called()
    assert codec.module._parameters["weight"] is original_weight
    assert codec.module._buffers["positions"] is original_buffer


@pytest.mark.parametrize("capacity", [True, False, 1.5, "16", None, 0, -1, 4097])
@pytest.mark.parametrize("has_ready_plan", [False, True])
def test_prepare_rejects_invalid_capacity_before_resuming_or_mutating_runtime(
    tmp_path, capacity, has_ready_plan
):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = SimpleNamespace(
        config=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=4096)),
        cache_capacity=4096,
    )
    engine.manager = SimpleNamespace(release_runtime_buffers=Mock())
    engine._resume_model = Mock()
    engine._release_graphs = Mock()
    engine._parked = True
    plan = engine._ready_plan = object() if has_ready_plan else None
    graph = engine.graph = object()
    backbone_graph = engine.backbone_graph = object()
    lifecycle = dict(engine._lifecycle)

    with pytest.raises(ValueError, match="outside the model's supported context"):
        engine._prepare(capacity)

    engine._resume_model.assert_not_called()
    engine.manager.release_runtime_buffers.assert_not_called()
    engine._release_graphs.assert_not_called()
    assert engine._lifecycle == lifecycle
    assert engine._ready_plan is plan
    assert engine.graph is graph
    assert engine.backbone_graph is backbone_graph
    assert engine._parked


@pytest.mark.parametrize("capacity", [1, 4096])
def test_prepare_accepts_supported_capacity_boundaries(tmp_path, capacity):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.model = SimpleNamespace(
        config=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=4096)),
        cache_capacity=4096,
    )
    engine.manager = SimpleNamespace(release_runtime_buffers=Mock())
    engine._resume_model = Mock()
    plan = engine._ready_plan = object()

    assert engine._prepare(capacity) == (plan, None)
    engine._resume_model.assert_called_once_with()
    engine.manager.release_runtime_buffers.assert_not_called()
    assert engine._lifecycle["runtime_reuses"] == 1
    assert engine._lifecycle["runtime_rebuilds"] == 0
