"""Local UI input validation, voice storage and path confinement."""

from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace

import pytest
import torch

import ui.server as server


@pytest.fixture(autouse=True)
def synthetic_worker_runtime(monkeypatch):
    # Native job tests replace inference with a test engine, not a local binary.
    monkeypatch.setattr(server, "load_native_runtime", lambda _root: {"protocol": 4, "sha256": "synthetic-runtime"})


class NativeTestEngine:
    is_gguf = True

    def __init__(self):
        self._lock = RLock()
        self.model_path = Path("test.gguf")
        self.options = SimpleNamespace(codec_max_frames=2048)

    def complete(self, text, output):
        from drama_low_vram.gguf_takes import save_native_take
        from drama_low_vram.protocols import PreviousTake
        from drama_low_vram.codec import write_wav
        codes = torch.zeros((10, 2), dtype=torch.int64)
        take = save_native_take(output, PreviousTake(text, codes))
        write_wav(torch.zeros(4096), output)
        return {"status": "complete", "output": str(output), "take_file": str(take),
                "audio": {"samples": 4096, "sample_rate": 44100}}


def test_job_timing_separates_cold_load_and_does_not_repeat_it_when_warm(tmp_path, monkeypatch):
    from drama_low_vram.job_timing import JobClock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    now = [0.0]
    monkeypatch.setattr(server, "JobClock", lambda **kwargs: JobClock(**kwargs, clock=lambda: now[0]))
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "model.gguf"
    state.checkpoint.write_bytes(b"synthetic")
    current = []

    class TimedNative(NativeTestEngine):
        loaded = False
        load_seconds = 18

        def generate(self, text, output, **kwargs):
            if not self.loaded:
                state._event(current[0], {"stage": "timing", "timing_phase": "model_load", "timing_action": "begin"})
                now[0] += 18
                assert state.job(current[0])["timing"]["generation_seconds"] == 0
                state._event(current[0], {"stage": "timing", "timing_phase": "model_load", "timing_action": "end"})
                self.loaded = True
            now[0] += 7
            return self.complete(text, output)

    engine = TimedNative()
    state._get_engine = lambda _: engine
    try:
        for expected_load in (18, 0):
            payload = {"mode": "tts", "text": "Timing fixture."}
            current[:] = [state.create_job(payload)]
            state._run_job(current[0], payload)
            job = state.job(current[0])
            assert job["status"] == "complete", job.get("error")
            assert job["timing"]["generation_seconds"] == 7
            assert job["timing"]["model_load_seconds"] == expected_load
            assert job["timing"]["phase"] == "finished"
            assert job["id"] not in state._job_clocks
            restored = server.AppState(data_root=state.data_root)
            assert restored.job(current[0])["timing"] == job["timing"]
            restored.shutdown()
    finally:
        state.shutdown()

class DeferredExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))
        return SimpleNamespace(cancel=lambda: True)

    def shutdown(self, **_kwargs):
        pass


def defer_workers(state):
    state._executor.shutdown(wait=True, cancel_futures=True)
    state._executor = DeferredExecutor()
    return state._executor


def test_reference_warmup_coalesces_duplicate_requests_and_can_be_cancelled(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "is_gguf", lambda _path: False)
    state = server.AppState()
    executor = defer_workers(state)
    state._engine = SimpleNamespace(close=lambda: None)
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"reference audio")
    state.checkpoint = tmp_path / "model"
    state._references = lambda payload: (
        [server.VoiceReference(str(audio), payload["transcript"], 0)]
        if payload.get("transcript") else [])
    try:
        first = state.start_reference_prewarm({"transcript": "Hello there."})
        duplicate = state.start_reference_prewarm({"transcript": "Hello there."})
        assert first["status"] == "queued"
        assert duplicate["id"] == first["id"]
        assert len(executor.calls) == 1

        cancelled = state.cancel_reference_prewarm()
        assert cancelled["status"] == "cancelled"
        empty = state.start_reference_prewarm({"transcript": ""})
        assert empty["status"] == "idle"
        assert state.reference_prewarm(first["id"])["status"] == "cancelled"
    finally:
        state.shutdown()


def test_gguf_reference_warmup_uses_native_preparation_without_creating_a_job(
        tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _root: {"protocol": 5})
    state = server.AppState()
    executor = defer_workers(state)
    state.checkpoint = tmp_path / "native.gguf"
    state.checkpoint.write_bytes(b"model")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    reference = server.VoiceReference(str(audio), "Reference.", 0)
    state._references = lambda _payload: [reference]
    engine = SimpleNamespace(
        is_gguf=True, status_callback=None, close=Mock(),
        prewarm_references=Mock(return_value={"status": "ready", "references": 1}),
        references_prepared=Mock(return_value=True),
    )
    state._engine = engine
    state._get_engine = lambda _identifier: engine
    try:
        queued = state.start_reference_prewarm({})
        assert queued["status"] == "queued"
        state._run_reference_prewarm(queued["id"])
        ready = state.reference_prewarm(queued["id"])
        assert ready["status"] == "ready"
        assert ready["result"]["references"] == 1
        engine.prewarm_references.assert_called_once()
        cancelled = engine.prewarm_references.call_args.kwargs["cancelled"]
        assert callable(cancelled) and not cancelled()
        assert state.start_reference_prewarm({})["id"] == queued["id"]
        assert len(executor.calls) == 1 and state.jobs() == []
    finally:
        state.shutdown()


def test_gguf_reference_warmup_stays_hidden_for_old_native_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _root: {"protocol": 4})
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "native.gguf"
    state.checkpoint.write_bytes(b"model")
    engine = state._engine = SimpleNamespace(is_gguf=True, loaded=True, close=lambda: None)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    state._references = lambda _payload: [server.VoiceReference(str(audio), "Reference.", 0)]
    try:
        assert state.start_reference_prewarm({}) == {"status": "unsupported"}
        assert state._engine is engine and state.jobs() == []
    finally:
        state.shutdown()


def test_reference_warmup_checks_residency_before_reusing_ready_status(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    executor = defer_workers(state)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    state._references = lambda _payload: [server.VoiceReference(str(audio), "Reference.")]
    engine = SimpleNamespace(
        status_callback=None, close=Mock(), reference_prefix_cached=Mock(return_value=True),
        prewarm_reference_prefix=Mock(return_value={"status": "ready", "_cache_key": "snapshot"}))
    state._engine = engine
    state._get_engine = lambda _id: engine
    try:
        first = state.start_reference_prewarm({})
        state._run_reference_prewarm(first["id"])
        ready = state.reference_prewarm(first["id"])
        assert ready["status"] == "ready" and "_cache_key" not in ready["result"]
        assert state.start_reference_prewarm({})["id"] == first["id"]
        assert len(executor.calls) == 1 and state.jobs() == []
        engine.reference_prefix_cached.return_value = False
        second = state.start_reference_prewarm({})
        assert second["id"] != first["id"] and len(executor.calls) == 2
        assert engine.status_callback is None
        state.cancel_reference_prewarm(first["id"])
        assert state.reference_prewarm(second["id"])["status"] == "queued"
    finally:
        state.shutdown()


def test_queued_warmup_never_loads_a_stale_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    state._engine = SimpleNamespace(close=lambda: None)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    state._references = lambda _payload: [server.VoiceReference(str(audio), "Reference.")]
    state._get_engine = lambda _id: pytest.fail("The old model must not be loaded")
    try:
        first = state.start_reference_prewarm({})
        state.checkpoint = tmp_path / "other-model"
        state._run_reference_prewarm(first["id"])
        assert state.reference_prewarm(first["id"])["status"] == "cancelled"
    finally:
        state.shutdown()


def test_restored_reference_does_not_cold_load_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    executor = defer_workers(state)
    state._references = lambda _payload: pytest.fail("Cold reference preparation must be deferred")
    state._get_engine = lambda _job: pytest.fail("Opening a page must not load an engine")
    try:
        assert state.start_reference_prewarm({"voice_ids": ["restored-voice"]}) == {"status": "idle"}
        assert state._engine is None and not executor.calls
    finally:
        state.shutdown()


def test_first_generation_prepares_reference_once_and_release_defers_it_again(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _root: {"protocol": 5})
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "model.gguf"
    state.checkpoint.write_bytes(b"synthetic")
    reference = server.VoiceReference(str(tmp_path / "reference.wav"), "Reference.")
    state._references = lambda _payload: [reference]
    calls = []

    class FirstUseEngine(NativeTestEngine):
        build = {"protocol": 5}
        loaded = False

        def prewarm_references(self, references, **kwargs):
            assert references == [reference]
            self.loaded = True
            calls.append("prepare")
            return {"status": "ready"}

        def generate(self, text, output, **kwargs):
            assert self.loaded
            calls.append("generate")
            return self.complete(text, output)

        def close(self):
            self.loaded = False

    def get_engine(_job):
        if state._engine is None:
            state._engine = FirstUseEngine()
        return state._engine

    state._get_engine = get_engine
    try:
        assert state.start_reference_prewarm({}) == {"status": "idle"}
        for _ in range(2):
            payload = {"mode": "tts", "text": "First use reference test."}
            job_id = state.create_job(payload)
            state._run_job(job_id, payload)
            assert state.job(job_id)["status"] == "complete", state.job(job_id).get("error")
        assert calls == ["prepare", "generate", "generate"]
        state._release_engine_resources()
        assert state.start_reference_prewarm({}) == {"status": "idle"}
        payload = {"mode": "tts", "text": "Run again after release."}
        job_id = state.create_job(payload)
        state._run_job(job_id, payload)
        assert state.job(job_id)["status"] == "complete", state.job(job_id).get("error")
        assert calls == ["prepare", "generate", "generate", "prepare", "generate"]
    finally:
        state.shutdown()


def test_release_closes_engine_and_preserves_models_history_and_references(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    engine = state._engine = SimpleNamespace(close=Mock())
    checkpoint = state.checkpoint
    state._jobs["saved"] = {"status": "complete", "output": "keep.wav"}
    voice = state.voices_root / "saved-reference.wav"
    voice.write_bytes(b"reference")
    try:
        result = state.release_resources()
        engine.close.assert_called_once()
        assert state._engine is None
        assert not result["engine_loaded"] and not result["releasing_resources"]
        assert state.checkpoint == checkpoint and state._jobs["saved"]["output"] == "keep.wav"
        assert voice.read_bytes() == b"reference"
        assert state.start_reference_prewarm({}) == {"status": "idle"}
        state.release_resources()
        engine.close.assert_called_once()
    finally:
        state.shutdown()


def test_python_first_use_keeps_reference_preparation_inside_the_generation_task(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    reference = server.VoiceReference(str(tmp_path / "reference.wav"), "Reference.")
    state._references = lambda _payload: [reference]
    calls = []
    engine = SimpleNamespace(
        load=lambda: calls.append("load"),
        close=Mock(),
        prewarm_reference_prefix=lambda references, **_kwargs: calls.append(("prepare", references)),
    )

    def get_engine(_job):
        state._engine = engine
        return engine

    def run(_mode, _text, output, **kwargs):
        assert kwargs["references"] == [reference]
        calls.append("generate")
        return NativeTestEngine().complete("Reference test.", output)

    state._get_engine = get_engine
    state._previous = lambda _payload, _engine: []
    monkeypatch.setattr(server, "DramaWorkflows", lambda _engine: SimpleNamespace(run=run))
    try:
        assert state.start_reference_prewarm({}) == {"status": "idle"}
        assert calls == []
        for _ in range(2):
            payload = {"mode": "tts", "text": "Reference test."}
            job_id = state.create_job(payload)
            state._run_job(job_id, payload)
            assert state.job(job_id)["status"] == "complete", state.job(job_id).get("error")
        assert calls == ["load", ("prepare", [reference]), "generate", "load", "generate"]
    finally:
        state.shutdown()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_release_does_not_interrupt_generation(tmp_path, monkeypatch, status):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    engine = state._engine = SimpleNamespace(close=Mock())
    state._jobs["current"] = {"status": status}
    try:
        with pytest.raises(RuntimeError, match="Finish or cancel generation"):
            state.release_resources()
        engine.close.assert_not_called()
        assert state._engine is engine and not state._releasing_resources
    finally:
        state._jobs.clear()
        state.shutdown()


def test_releasing_blocks_new_work_and_waits_for_preparation_cancellation(tmp_path, monkeypatch):
    from threading import Thread
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    engine = state._engine = SimpleNamespace(close=Mock())
    started, proceed = Event(), Event()
    results = []

    def prepare():
        started.set()
        proceed.wait(timeout=5)

    state._executor.submit(prepare)
    assert started.wait(timeout=5)
    thread = Thread(target=lambda: results.append(state.release_resources()), daemon=True)
    try:
        thread.start()
        import time
        deadline = time.monotonic() + 2
        while not state._releasing_resources and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state._releasing_resources
        with pytest.raises(RuntimeError, match="being released"):
            state.create_job({"mode": "tts", "text": "Do not start yet."})
        assert state.start_reference_prewarm({}) == {"status": "idle"}
        engine.close.assert_not_called()
        proceed.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(results) == 1
        engine.close.assert_called_once()
        assert not state._releasing_resources and state._engine is None
    finally:
        proceed.set()
        thread.join(timeout=5)
        state.shutdown()


def test_gguf_queue_dispatches_plain_tts_without_python_workflows(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "native.gguf"
    state.checkpoint.write_bytes(b"model")
    calls = []
    class FakeNative(NativeTestEngine):
        def generate(self, text, output, **kwargs):
            calls.append((text, kwargs))
            return self.complete(text, output)
    state._get_engine = lambda _: FakeNative()
    try:
        job = state.create_job({"mode": "tts", "text": "Hello", "max_new_tokens": 215})
        assert state.job(job)["recipe"]["top_k"] == 0
        state._run_job(job, {"mode": "tts", "text": "Hello", "max_new_tokens": 215})
        assert state.job(job)["status"] == "complete"
        assert calls[0][0] == "Hello" and calls[0][1]["max_frames"] == 215
        assert state.job(job)["take_file"].endswith(".take.json")
        with pytest.raises(ValueError, match="source take"):
            state.create_job({"mode": "infill", "text": "Hello"})
        payload = {"mode": "tts", "text": "Hello", "direction": "Calm",
                   "language": "zh", "duration_frames": 215}
        controlled = state.create_job(payload)
        state._run_job(controlled, payload)
        assert state.job(controlled)["status"] == "complete"
        assert calls[-1][1]["duration_frames"] == 215
        assert calls[-1][1]["direction"] == "Calm" and calls[-1][1]["language"] == "zh"
    finally:
        state.shutdown()


def test_gguf_cancel_result_is_recorded_as_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    class FakeNative(NativeTestEngine):
        def generate(self, *_args, **_kwargs):
            raise server.GenerationCancelled()
    state._get_engine = lambda _: FakeNative()
    try:
        job = state.create_job({"mode": "tts", "text": "Hello"})
        state._run_job(job, {"mode": "tts", "text": "Hello"})
        assert state.job(job)["status"] == "cancelled"
        assert state.job(job)["error"] is None
    finally:
        state.shutdown()


@pytest.mark.parametrize("outcome", ["complete", "error", "cancelled"])
def test_worker_console_is_concise_but_keeps_success_errors_and_cancellation(tmp_path, monkeypatch, outcome):
    import io
    import sys
    from drama_low_vram.job_timing import JobClock
    from drama_low_vram.studio_logging import console_session

    monkeypatch.setattr(server, "ROOT", tmp_path)
    now = [0.0]
    monkeypatch.setattr(server, "JobClock", lambda **kwargs: JobClock(**kwargs, clock=lambda: now[0]))
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "fixture.gguf"

    class Native(NativeTestEngine):
        def generate(self, text, output, **_kwargs):
            print("Verbose native backend setup.")
            print("Per-layer diagnostic detail.", file=sys.stderr)
            for frame in range(100):
                state._event(identifier, {"stage": "generate", "frame": frame})
            now[0] += 7
            if outcome == "error":
                raise RuntimeError("Synthetic worker failure\nAdditional internal detail")
            if outcome == "cancelled":
                raise server.GenerationCancelled()
            return self.complete(text, output)

    state._get_engine = lambda _: Native()
    console = io.StringIO()
    try:
        with console_session(tmp_path, console=console) as log:
            payload = {"mode": "tts", "text": "Console fixture."}
            identifier = state.create_job(payload)
            state._run_job(identifier, payload)
        shown = console.getvalue()
        assert state.job(identifier)["status"] == outcome
        assert "started." in shown and len(shown.splitlines()) == 2
        assert "Verbose" not in shown and "Per-layer" not in shown and "Traceback" not in shown
        saved = log.path.read_text(encoding="utf-8")
        assert "Verbose native backend setup." in saved and "Per-layer diagnostic detail." in saved
        if outcome == "complete":
            assert "Generation: 7.000 s." in shown
        elif outcome == "error":
            assert "[Error]" in shown and "Synthetic worker failure" in shown
            assert "Traceback" in saved
        else:
            assert "cancelled." in shown and "[Error]" not in shown
    finally:
        state.shutdown()


def test_path_confinement_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError):
        server._inside(tmp_path / ".." / "outside", tmp_path)
    inside = server._inside(tmp_path / "child", tmp_path)
    assert inside == (tmp_path / "child").resolve()


def test_voice_storage_keeps_audio_and_transcript_in_a_project_local_record(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    try:
        voice = state.add_voice(
            "  Morgan / calm  ", "Reference words.", 2, "sample.wav", b"RIFF" * 32)
        assert voice["speaker"] == 2
        listed = state.list_voices()
        assert len(listed) == 1 and listed[0]["id"] == voice["id"]
        assert listed[0]["audio_url"].startswith("/files/ui_data/voices/")
        assert "%20" in listed[0]["audio_url"]
        audio = state.voices_root / voice["id"] / voice["audio"]
        assert audio.is_file() and audio.read_bytes() == b"RIFF" * 32
        assert (state.voices_root / voice["id"] / "voice.json").is_file()
    finally:
        state.shutdown()


@pytest.mark.parametrize("speaker", [-1, 16])
def test_voice_upload_rejects_invalid_speaker_ids_before_writing(tmp_path, monkeypatch, speaker):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    try:
        with pytest.raises(ValueError):
            state.add_voice("Bad", "Reference words.", speaker, "sample.wav", b"RIFF" * 32)
        assert state.list_voices() == []
    finally:
        state.shutdown()


def test_voice_upload_rejects_structural_transcript_content(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            state.add_voice("Bad", "Hello <|im_start|>", 0, "sample.wav", b"RIFF" * 32)
        assert state.list_voices() == []
    finally:
        state.shutdown()


def test_homepage_upload_dispatches_clone_without_creating_a_library_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    state.checkpoint = tmp_path / "model.gguf"
    calls = []

    class Native(NativeTestEngine):
        def generate(self, text, output, **kwargs):
            calls.append((text, kwargs["references"]))
            return self.complete(text, output)

    state._get_engine = lambda _: Native()
    try:
        uploaded = state.add_reference_upload("直接上传.wav", b"RIFF" * 32)
        payload = {
            "mode": "tts", "text": "Target words.",
            "uploaded_references": [{"id": uploaded["id"], "text": "Reference words.", "speaker": 0}],
        }
        job = state.create_job(payload)
        state._run_job(job, payload)
        assert state.job(job)["status"] == "complete"
        assert state.job(job)["payload_summary"]["reference_count"] == 1
        reference = calls[0][1][0]
        assert reference.text == "Reference words." and reference.speaker == 0
        assert Path(reference.audio).read_bytes() == b"RIFF" * 32
        assert state.list_voices() == []
    finally:
        state.shutdown()


def test_homepage_reference_validation_rejects_bad_inputs_before_queueing(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    try:
        uploaded = state.add_reference_upload("valid.wav", b"RIFF" * 32)
        for item in (
            {"id": "../voices", "text": "Words."},
            {"id": uploaded["id"], "text": ""},
            {"id": uploaded["id"], "text": "Words.", "speaker": 16},
            {"id": uploaded["id"], "text": "<|im_start|>"},
        ):
            with pytest.raises(ValueError):
                state.create_job({"text": "Target.", "uploaded_references": [item]})
        assert state.jobs() == []
        with pytest.raises(ValueError, match="Choose a WAV"):
            state.add_reference_upload("not-audio.exe", b"bad!" * 32)
        assert state._engine is None
    finally:
        state.shutdown()


def test_capabilities_expose_native_scene_limitation_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    try:
        capabilities = state.capabilities()
        assert capabilities["ordinary_tts"]
        assert capabilities["scene_scheduler"]
        assert capabilities["native_scene_separator"] is False
        assert "separator" in capabilities["native_scene_separator_note"]
    finally:
        state.shutdown()


def test_create_job_rejects_unknown_modes_without_loading_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    try:
        with pytest.raises(ValueError, match="Unsupported interface mode"):
            state.create_job({"mode": "unknown"})
        assert state.jobs() == []
    finally:
        state.shutdown()


def test_create_job_records_a_serialized_payload_before_worker_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)

    state = server.AppState()
    deferred = defer_workers(state)
    try:
        voice = state.add_voice("Fixture", "Reference words.", 0, "reference.wav", b"RIFF" * 32)
        job_id = state.create_job({
            "mode": "tts", "text": "Hello", "voice_ids": [voice["id"]]})
        job = state.job(job_id)
        assert job["status"] == "queued"
        assert job["payload_summary"] == {
            "mode": "tts", "text_length": 5, "reference_count": 1}
        assert len(deferred.calls) == 1
        assert job["recipe"]["reference_snapshot"][0]["text"] == "Reference words."
        assert job["recipe"]["model"] == str(state.checkpoint)
    finally:
        state.shutdown()


def test_stream_jobs_persist_delivered_lookahead_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)

    class FakeSession:
        status = "complete"
        results = []
        pending = 0

        def __init__(self, *_args, **_kwargs):
            self.rows = [
                {"index": 0, "text": "One.", "lookahead": "Two.", "take_file": "one.take"},
                {"index": 1, "text": "Two.", "lookahead": "", "take_file": "two.take"},
            ]

        def push(self, _text):
            pass

        def finish(self):
            pass

        def cancel(self):
            self.status = "cancelled"

        def run(self, **_kwargs):
            yield from self.rows

        def export(self, output):
            Path(output).write_bytes(b"fake-wave")
            return Path(output)

    monkeypatch.setattr(server, "TextStreamSession", FakeSession)
    monkeypatch.setattr(server, "DramaWorkflows", lambda _engine: object())
    state = server.AppState()
    defer_workers(state)
    try:
        state._get_engine = lambda _job_id: SimpleNamespace(
            model=SimpleNamespace(), load=lambda: None)
        job_id = state.create_job({"mode": "stream", "chunks": ["One.", "Two."]})
        state._run_job(job_id, {"mode": "stream", "chunks": ["One.", "Two."]})
        job = state.job(job_id)
        assert job["status"] == "complete"
        assert [row["lookahead"] for row in job["result"]["segments"]] == ["Two.", ""]
        assert job["result"]["exported_pcm_matches_delivered_chunks"] is True
    finally:
        state.shutdown()


def test_persisted_running_job_is_recovered_as_interrupted_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    jobs_root = tmp_path / "ui_data" / "jobs" / "old-job"
    jobs_root.mkdir(parents=True)
    (jobs_root / "job.json").write_text(
        '{"id":"old-job","status":"running","created_at":1,"updated_at":2,"events":[]}',
        encoding="utf-8")
    state = server.AppState()
    try:
        job = state.job("old-job")
        assert job["status"] == "interrupted"
    finally:
        state.shutdown()


def test_infill_job_does_not_forward_revision_history_or_duration_to_workflow(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    calls = []

    class FakeWorkflows:
        def run(self, mode, text, output, **kwargs):
            calls.append((mode, text, output, kwargs))
            return {"status": "incomplete", "output": None, "take_file": None}

    fake_engine = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        load=lambda: None,
    )
    monkeypatch.setattr(server, "DramaWorkflows", lambda _engine: FakeWorkflows())
    monkeypatch.setattr(
        server, "load_take",
        lambda *_args: (SimpleNamespace(codes=torch.zeros(3, 8, dtype=torch.int64)), {}, None),
    )
    state = server.AppState()
    defer_workers(state)
    state._get_engine = lambda _job_id: fake_engine
    try:
        job_id = state.create_job({"mode": "infill", "text": "Edited."})
        state._run_job(job_id, {
            "mode": "infill",
            "text": "Edited.",
            "source_take": str(tmp_path / "source.take.npz"),
            "previous_take": str(tmp_path / "previous.take.npz"),
            "start_frame": 2,
            "end_frame": 5,
            "duration_frames": 4,
        })
        assert calls
        _mode, _text, _output, kwargs = calls[0]
        assert "previous_takes" not in kwargs
        assert "duration_frames" not in kwargs
        assert "infill" in kwargs
    finally:
        state.shutdown()


def test_live_stream_accepts_incremental_input_and_seals_its_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    try:
        job_id = state.create_job({"mode": "stream", "live_stream": True, "chunks": ["One."]})
        session = state._streams[job_id]
        assert session.pending == 1 and state.job(job_id)["stream_input"]["live"]
        state.stream_input(job_id, text="Two.")
        assert session.pending == 2
        state.stream_input(job_id, finish=True)
        assert state.job(job_id)["stream_input"]["finished"]
        with pytest.raises(ValueError, match="no longer accepts"):
            state.stream_input(job_id, text="Late.")
    finally:
        state.shutdown()


@pytest.mark.parametrize("payload", [
    {"mode": "stream", "chunks": []},
    {"mode": "stream", "chunks": "Not a list"},
    {"mode": "stream", "chunks": ["One."], "live_stream": "yes"},
    {"mode": "stream", "chunks": [123]},
    {"mode": "stream", "chunks": ["One."], "direction": "This field would be ignored"},
])
def test_invalid_stream_input_never_enqueues_a_model_job(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    deferred = defer_workers(state)
    try:
        with pytest.raises((TypeError, ValueError)):
            state.create_job(payload)
        assert not deferred.calls and not state.jobs()
    finally:
        state.shutdown()


def test_cancelled_queued_job_never_loads_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    defer_workers(state)
    state._get_engine = lambda _id: pytest.fail("Cancelled queued work must not load weights")
    try:
        job_id = state.create_job({"mode": "tts", "text": "Hello."})
        state.cancel(job_id)
        state._run_job(job_id, {"mode": "tts", "text": "Hello."})
        assert state.job(job_id)["status"] == "cancelled"
        assert state.job(job_id)["output"] is None
    finally:
        state.shutdown()


def test_cancel_wakes_a_live_consumer_and_releases_the_serial_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    waiting, completed_next = Event(), Event()

    class Workflows:
        def run(self, _mode, _text, _output, **_kwargs):
            completed_next.set()
            return {"status": "complete", "output": None}

    monkeypatch.setattr(server, "DramaWorkflows", lambda _engine: Workflows())
    state = server.AppState()
    state._get_engine = lambda _id: SimpleNamespace(model=SimpleNamespace(), load=lambda: None)
    original_event = state._event

    def event(job_id, value):
        original_event(job_id, value)
        if value["stage"] == "waiting_for_text":
            waiting.set()

    state._event = event
    try:
        first = state.create_job({"mode": "stream", "live_stream": True, "chunks": ["Waiting."]})
        assert waiting.wait(timeout=5)
        second = state.create_job({"mode": "tts", "text": "Next."})
        assert state.job(second)["status"] == "queued"
        state.cancel(first)
        assert completed_next.wait(timeout=5)
        state._executor.shutdown(wait=True)
        assert state.job(first)["status"] == "cancelled"
        assert state.job(first)["output"] is None
        assert state.job(second)["status"] == "complete"
    finally:
        state.shutdown()


def test_model_selection_is_persistent_and_closes_only_an_idle_previous_engine(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "inspect_checkpoint",
                        lambda directory: {"directory": str(directory), "ready": True})
    state = server.AppState()
    old = state._engine = SimpleNamespace(close=Mock())
    selected = tmp_path / "different-drive-model"
    try:
        status = state.select_checkpoint(str(selected))
        old.close.assert_called_once()
        assert state._engine is None
        assert Path(status["selected_directory"]) == selected.resolve()
        assert state.expert_cache_directory().is_relative_to(tmp_path / ".cache")
        restored = server.AppState()
        try:
            assert restored.checkpoint == selected.resolve()
            assert restored.expert_cache_directory() == state.expert_cache_directory()
        finally:
            restored.shutdown()
    finally:
        state.shutdown()


def test_model_directory_picker_is_limited_to_the_local_same_origin_page():
    allowed = server._local_model_picker_request(
        "127.0.0.1", "127.0.0.1:8879", "http://127.0.0.1:8879", 8879)
    assert allowed
    assert server._local_model_picker_request(
        "::1", "[::1]:8879", "http://[::1]:8879", 8879)
    assert not server._local_model_picker_request(
        "192.168.1.20", "127.0.0.1:8879", "http://127.0.0.1:8879", 8879)
    assert not server._local_model_picker_request(
        "127.0.0.1", "127.0.0.1:8879", "http://example.com:8879", 8879)
    assert not server._local_model_picker_request(
        "127.0.0.1", "example.com:8879", "http://example.com:8879", 8879)


def test_relative_model_selection_follows_a_moved_distribution(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    monkeypatch.setattr(server, "ROOT", root)
    monkeypatch.setattr(server, "inspect_checkpoint", lambda path: {"directory": str(path), "ready": True})
    state = server.AppState()
    try:
        state.select_checkpoint(str(root / "checkpoints/local"))
        saved = state.settings_file.read_bytes()
        import json
        assert not Path(json.loads(saved)["checkpoint"]).is_absolute()
    finally:
        state.shutdown()
    moved = tmp_path / "moved distribution"
    (moved / "ui_data").mkdir(parents=True)
    (moved / "ui_data/settings.json").write_bytes(saved)
    monkeypatch.setattr(server, "ROOT", moved)
    restored = server.AppState()
    try:
        assert restored.checkpoint == (moved / "checkpoints/local").resolve()
    finally:
        restored.shutdown()


def test_model_catalog_discovers_custom_folders_and_arbitrary_gguf_names(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints"
    (checkpoint / "custom-precision").mkdir(parents=True)
    (checkpoint / "custom-precision/config.json").write_text("{}")
    (checkpoint / "more-gguf").mkdir()
    native = checkpoint / "more-gguf/renamed-model-q5.gguf"
    native.write_bytes(b"catalog does not load weights")
    root_native = checkpoint / "another-model-q4.gguf"
    root_native.write_bytes(b"catalog does not load weights")
    state = server.AppState()
    try:
        rows = state.available_models()
        paths = {Path(row["directory"]) for row in rows}
        assert {native, root_native, checkpoint / "custom-precision", state.checkpoint} <= paths
        assert len(rows) == len(paths)
        native_row = next(row for row in rows if row["directory"] == str(native))
        assert native_row["runtime"] == "gguf"
        assert native_row["name"] == "more-gguf"
        assert native_row["location"] == str(native.parent)
        assert next(row for row in rows if row["directory"] == str(root_native))["location"] == str(checkpoint)
        assert state._engine is None
    finally:
        state.shutdown()


def test_external_model_choices_survive_switching_and_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "inspect_checkpoint",
                        lambda directory: {"directory": str(directory), "ready": True})
    paths = [tmp_path / "external-a", tmp_path / "external-b"]
    for path in paths:
        path.mkdir()
    state = server.AppState()
    try:
        for path in paths:
            state.select_checkpoint(path)
    finally:
        state.shutdown()
    restored = server.AppState()
    try:
        assert restored.checkpoint == paths[-1]
        listed = {Path(row["directory"]) for row in restored.available_models()}
        assert set(paths) <= listed
        assert restored._engine is None
    finally:
        restored.shutdown()


def test_model_status_distinguishes_preview_from_selection_and_reports_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    selected = tmp_path / "checkpoints/drama-3"
    def inspect(path):
        return {
            "directory": str(path), "ready": Path(path) == selected,
            "storage_format": "FP8" if Path(path) == selected else "other",
        }
    monkeypatch.setattr(server, "inspect_checkpoint", inspect)
    state = server.AppState()
    defer_workers(state)
    try:
        state.create_job({"mode": "tts", "text": "Queued."})
        status = state.model_status(tmp_path / "not-ready-preview")
        assert status["busy"] is True
        assert status["selected_ready"] is True
        assert status["inspection"]["ready"] is False
        assert status["selected_format"] == "FP8"
        assert state._engine is None
    finally:
        state.shutdown()


def test_model_switch_during_queued_generation_is_rejected_without_changing_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "inspect_checkpoint",
                        lambda directory: {"directory": str(directory), "ready": True})
    state = server.AppState()
    defer_workers(state)
    original = state.checkpoint
    try:
        state.create_job({"mode": "tts", "text": "Queued."})
        with pytest.raises(RuntimeError, match="Finish or cancel"):
            state.select_checkpoint(str(tmp_path / "other-model"))
        assert state.checkpoint == original
        assert not state.settings_file.exists()
    finally:
        state.shutdown()


def test_selected_external_model_keeps_runtime_caches_on_the_project_drive(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "inspect_checkpoint", lambda _directory: {"ready": True})
    runtime = SimpleNamespace(close=Mock())
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(server, "DramaEngine", constructor)
    host_threads = Mock()
    monkeypatch.setattr("torch.set_num_threads", host_threads)
    state = server.AppState()
    state.checkpoint = tmp_path.parent / "external-checkpoint"
    try:
        state._get_engine("diagnostic")
        options = constructor.call_args.args[0]
        assert options.checkpoint == state.checkpoint
        assert options.expert_cache_dir.is_relative_to(tmp_path / ".cache")
        assert options.compute_profile_dir == tmp_path / ".cache/compute-profiles"
        assert options.cpu_experts == 0
        assert options.staged_experts is None and options.prefill_chunk is None
        host_threads.assert_called_once_with(1)
    finally:
        state.shutdown()


def test_take_inspection_recovers_transcript_and_duration_without_loading_weights(tmp_path, monkeypatch):
    import json
    from drama_low_vram.workflows import save_take

    monkeypatch.setattr(server, "ROOT", tmp_path)
    config = {"semantic_start_token_id": 248099}
    model_config = SimpleNamespace(
        audio_decoder=SimpleNamespace(num_codebooks=10, vocab_size=4096))
    monkeypatch.setattr(
        "fish_speech.models.text2semantic.fish_omni.config.FishOmniConfig.from_dict",
        lambda _document: model_config)
    checkpoint = tmp_path / "checkpoints/drama-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    codes = torch.arange(10 * 45).reshape(10, 45) % 4096
    frames = torch.cat((codes[:1] + config["semantic_start_token_id"], codes)).T
    output = tmp_path / "source.wav"
    output.write_bytes(b"saved source audio")
    take = save_take(output, "Actual complete transcript.", codes, frames, metadata={"mode": "tts"})
    state = server.AppState()
    state._get_engine = lambda _id: pytest.fail("Editing metadata must not load GPU weights")
    try:
        value = state.inspect_take(str(take))
        assert value["text"] == "Actual complete transcript."
        assert value["frames"] == 45
        assert value["duration_seconds"] == 45 * 2048 / 44100
        assert value["audio"] == str(output)
        assert state._engine is None
        with pytest.raises(ValueError, match="outside"):
            state.inspect_take(str(tmp_path.parent / "outside.take.npz"))
    finally:
        state.shutdown()
