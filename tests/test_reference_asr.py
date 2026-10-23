"""Reference transcription cache, queue serialization and input confinement."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from drama_low_vram.reference_asr import (
    MODEL_FILES, LocalASRUnavailable, ReferenceTranscriber, ensure_model,
)
import ui.server as server


def test_transcription_reuses_audio_content_across_uploads_and_restart(tmp_path):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    first.write_bytes(b"same speech audio")
    second.write_bytes(first.read_bytes())
    calls = []

    def runner(audio, model_dir, root, language, progress, **_kwargs):
        calls.append((audio, language))
        progress("transcribing")
        return {"text": "Actual words.", "language": "English", "device": "cuda:0"}

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = ReferenceTranscriber(tmp_path, tmp_path, executor, runner)
        task = worker.start(first)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(task["id"])["status"] == "complete"
        restored = ReferenceTranscriber(tmp_path, tmp_path, executor, runner)
        cached = restored.start(second)
        assert cached["cache_hit"] and cached["result"]["text"] == "Actual words."
        assert len(calls) == 1
        retry = restored.start(second, force=True)
        executor.submit(lambda: None).result(timeout=5)
        assert not restored.get(retry["id"])["cache_hit"] and len(calls) == 2
        restored.start(second, language="zh")
        executor.submit(lambda: None).result(timeout=5)
        assert calls[-1][1] == "Chinese" and len(calls) == 3


def test_pending_transcription_is_deduplicated_and_serialized_with_tts_queue(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    held, started = threading.Event(), threading.Event()
    calls = []

    def runner(*_args, **_kwargs):
        calls.append("asr")
        return {"text": "Words."}

    def tts():
        started.set()
        held.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(tts)
        assert started.wait(timeout=5)
        worker = ReferenceTranscriber(tmp_path, tmp_path, executor, runner)
        first = worker.start(audio)
        duplicate = worker.start(audio)
        assert duplicate["id"] == first["id"] and calls == []
        assert first["status"] == "queued"
        held.set()
        executor.submit(lambda: None).result(timeout=5)
        assert calls == ["asr"]


def test_transcription_failure_does_not_poison_cache_and_can_retry(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    with ThreadPoolExecutor(max_workers=1) as executor:
        def failure(*_args, **_kwargs):
            raise RuntimeError("Test inference failure")
        worker = ReferenceTranscriber(tmp_path, tmp_path, executor, failure)
        task = worker.start(audio)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(task["id"])["status"] == "error"
        assert not list((tmp_path / "transcripts").glob("*.json"))
        worker.runner = lambda *_args, **_kwargs: {"text": "Recovered."}
        retry = worker.start(audio)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(retry["id"])["status"] == "complete"


def test_transcription_resolves_only_registered_reference_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    calls = []
    state.asr.runner = lambda audio, *_args, **_kwargs: (
        calls.append(str(audio)) or {"text": "Recognized words.", "language": "English"})
    try:
        uploaded = state.add_reference_upload("sample.wav", b"RIFF" * 32)
        task = state.start_transcription({"upload_id": uploaded["id"], "local_only": False})
        state._executor.submit(lambda: None).result(timeout=5)
        assert state.asr.get(task["id"])["result"]["text"] == "Recognized words."
        assert Path(calls[0]).is_relative_to(state.uploads_root)
        for payload in (
            {"upload_id": "../escape"},
            {"upload_id": uploaded["id"], "language": "invalid"},
            {"upload_id": uploaded["id"], "force": "false"},
            {"upload_id": uploaded["id"], "local_only": "true"},
            {"upload_id": uploaded["id"], "voice_id": "another"},
        ):
            with pytest.raises(ValueError):
                state.start_transcription(payload)
        assert state.list_voices() == []
        assert state.jobs() == []
    finally:
        state.shutdown()


def test_asr_checkpoint_is_not_offered_as_a_tts_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    folder = tmp_path / "checkpoints/Qwen3-ASR-0.6B-hf"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
    state = server.AppState()
    try:
        assert str(folder) not in {row["directory"] for row in state.available_models()}
    finally:
        state.shutdown()


def test_gpu_oom_falls_back_to_cpu_without_touching_tts(tmp_path, monkeypatch):
    import torch
    import drama_low_vram.reference_asr as asr

    devices, stages = [], []
    monkeypatch.setattr(asr, "read_reference", lambda *_args: ("samples", 3.0))
    monkeypatch.setattr(asr, "ensure_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (1, 2))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def recognize(samples, folder, device, language, progress):
        devices.append(device)
        if device.startswith("cuda"):
            raise torch.cuda.OutOfMemoryError("Simulated insufficient memory")
        return {"text": "Recovered on CPU.", "device": device}

    monkeypatch.setattr(asr, "_recognize", recognize)
    result = asr.transcribe_audio(tmp_path / "clip.wav", tmp_path, tmp_path, None, stages.append)
    assert devices == ["cuda:0", "cpu"]
    assert result["fallback"] == "gpu_out_of_memory"
    assert "cpu_fallback" in stages


def test_local_only_skips_missing_or_incomplete_model_without_queuing(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = ReferenceTranscriber(
            tmp_path, tmp_path, executor,
            lambda *_args, **_kwargs: calls.append("inference") or {"text": "Words."})
        worker.model_dir = tmp_path / "model"
        for complete in (False, True):
            if complete:
                worker.model_dir.mkdir()
                (worker.model_dir / "config.json").write_text("{}")
            for force in (False, True):
                task = worker.start(audio, local_only=True, force=force)
                assert task["status"] == "skipped"
                assert task["reason"] == "model_unavailable"
                assert "error" not in task
        executor.submit(lambda: None).result(timeout=5)
    assert calls == []
    assert not list((tmp_path / "transcripts").glob("*.json"))


def test_local_only_reads_cache_without_model_and_never_redownloads_on_retry(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = ReferenceTranscriber(
            tmp_path, tmp_path, executor,
            lambda *_args, **kwargs: calls.append(kwargs) or {"text": "Cached words."})
        worker.model_dir = tmp_path / "missing-model"
        task = worker.start(audio)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(task["id"])["status"] == "complete"
        cached = worker.start(audio, local_only=True)
        assert cached["cache_hit"] and cached["result"]["text"] == "Cached words."
        retry = worker.start(audio, local_only=True, force=True)
        assert retry["status"] == "skipped"
    assert calls == [{"local_only": False}]


def test_complete_local_model_runs_and_missing_during_queue_is_skipped(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    calls = []

    def runner(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise LocalASRUnavailable()
        return {"text": "Local words."}

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = ReferenceTranscriber(tmp_path, tmp_path, executor, runner)
        worker.model_dir = tmp_path / "model"
        worker.model_dir.mkdir()
        for name in MODEL_FILES:
            (worker.model_dir / name).write_bytes(b"fixture")
        task = worker.start(audio, local_only=True)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(task["id"])["result"]["text"] == "Local words."
        retry = worker.start(audio, local_only=True, force=True)
        executor.submit(lambda: None).result(timeout=5)
        assert worker.get(retry["id"])["status"] == "skipped"
    assert calls == [{"local_only": True}, {"local_only": True}]


def test_ensure_local_model_never_imports_download_or_reports_downloading(tmp_path):
    stages = []
    with pytest.raises(LocalASRUnavailable):
        ensure_model(tmp_path / "missing", tmp_path, stages.append, local_only=True)
    assert stages == []


def test_ui_transcription_defaults_to_optional_local_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    calls = []
    state.asr.model_dir = tmp_path / "missing-model"
    state.asr.runner = lambda *_args, **_kwargs: calls.append("inference")
    try:
        uploaded = state.add_reference_upload("sample.wav", b"RIFF" * 32)
        task = state.start_transcription({"upload_id": uploaded["id"]})
        assert task["status"] == "skipped" and "error" not in task
        state._executor.submit(lambda: None).result(timeout=5)
        assert calls == []
    finally:
        state.shutdown()
