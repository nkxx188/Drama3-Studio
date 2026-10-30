"""Uploaded audio is prepared inside its edit job; ordinary editing stays independent."""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import ui.server as server
from drama_low_vram.gguf_takes import load_conditioning_take


@pytest.fixture
def importer(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _: {"protocol": 5})
    state = server.AppState()
    state.checkpoint = tmp_path / "model.gguf"
    state.checkpoint.write_bytes(b"test model")
    tasks = []
    monkeypatch.setattr(
        state._executor, "submit",
        lambda function, *args: tasks.append((function, args)) or SimpleNamespace(cancel=lambda: True),
    )
    try:
        yield state, tasks
    finally:
        state.shutdown()


def recording(state, seconds=2, format="WAV"):
    samples = np.sin(np.arange(int(seconds * 24000)) * (2 * np.pi * 220 / 24000)) * .1
    buffer = io.BytesIO()
    sf.write(buffer, samples, 24000, format=format)
    return state.add_reference_upload("source." + format.lower(), buffer.getvalue())


def request(upload, operation="reword", **kwargs):
    return {
        "upload_id": upload["id"], "original_transcript": "Original words.",
        "text": "Original words." if operation == "revision" else "Updated words.",
        "operation": operation, "direction": "" if operation == "infill" else "Speak softly.",
        "start_frame": 8, "end_frame": 16, **kwargs,
    }


def run_next(tasks):
    function, args = tasks.pop(0)
    function(*args)


def test_uploaded_edit_automatically_budgets_length_and_preserves_retry_switch(importer):
    state, tasks = importer
    upload = recording(state)
    created = state.start_uploaded_audio_edit(request(upload, automatic_length=True, silence_retry=True))
    recipe = state.job(created["id"])["recipe"]
    assert recipe["max_new_tokens"] == 768 and recipe["length_policy"] == "automatic-v1"
    assert recipe["silence_retry"] is True and recipe["duration_frames"] is None
    for bad in ("true", 1, None):
        with pytest.raises(ValueError, match="silence_retry"):
            state.start_uploaded_audio_edit(request(upload, silence_retry=bad))


@pytest.mark.parametrize("backend,format", [("gguf", "WAV"), ("python", "FLAC")])
@pytest.mark.parametrize("operation", ["revision", "reword", "infill"])
def test_upload_is_prepared_only_with_edit_generation(importer, monkeypatch, backend, format, operation):
    state, tasks = importer
    if backend == "python":
        state.checkpoint = state.checkpoint.with_suffix("")
        monkeypatch.setattr(server, "inspect_checkpoint", lambda _: {"ready": True})
    encoded = torch.arange(10 * 44, dtype=torch.int64).reshape(10, 44)
    calls, generated = [], []

    class Encoder:
        is_gguf = backend == "gguf"
        build = {"protocol": 5}
        model_path = state.checkpoint

        def load(self, **kwargs):
            calls.append("load")

        def prepare_reference(self, path, transcript):
            assert Path(path).name == "source.wav" and transcript == "<|speaker:0|> Original words."
            calls.append("encode")
            return encoded, False

        def prewarm_references(self, references, **kwargs):
            assert len(references) == 1 and Path(references[0].audio).name == "source.wav"
            calls.append("encode")

        def _cached_reference_codes(self, path):
            return {"vq_codes": encoded.tolist()}

    state._get_engine = lambda _: Encoder()

    def generate(identifier, payload):
        generated.append(payload)
        job = state.job(identifier)
        source = job["source_preparation"]["take_file"]
        state._record_result(identifier, {
            "status": "complete", "output": str(Path(source).with_suffix(".wav")), "take_file": source,
        })

    state._run_job = generate
    upload = recording(state, format=format)
    original = Path(state.reference_upload(upload["id"])["audio_path"]).read_bytes()
    assert not tasks and not calls
    created = state.start_uploaded_audio_edit(request(upload, operation))
    assert not calls and not generated and len(tasks) == 1
    assert len(state.jobs()) == 1 and state.job(created["id"])["status"] == "queued"
    run_next(tasks)
    result = state.job(created["id"])
    assert result["status"] == "complete", result.get("error")
    assert calls == ["load", "encode"] and len(generated) == 1
    assert Path(state.reference_upload(upload["id"])["audio_path"]).read_bytes() == original
    assert result["studio"]["result_scope"] == "audio_editor"
    assert result["source_preparation"]["codec_cache_hit"] is False
    take, _, _ = load_conditioning_take(result["source_preparation"]["take_file"])
    assert torch.equal(take.codes, encoded)
    payload = generated[0]
    if operation == "infill":
        assert payload["source_take"] == result["source_preparation"]["take_file"]
        assert (payload["start_frame"], payload["end_frame"]) == (8, 16)
        assert payload["reference_snapshot"] == []
    elif operation == "revision":
        assert payload["previous_take"] == result["source_preparation"]["take_file"]
        assert payload["reference_snapshot"][0]["text"] == "Original words."
    else:
        assert payload["mode"] == "tts" and "previous_take" not in payload
        assert Path(payload["reference_snapshot"][0]["audio"]).is_file()

    # Corrected words do not alter audio codes or require encoding a second time.
    changed = request(upload, original_transcript="Corrected words.")
    if operation == "revision":
        changed["operation"] = "reword"
    second = state.start_uploaded_audio_edit(changed)
    run_next(tasks)
    assert state.job(second["id"])["status"] == "complete"
    assert state.job(second["id"])["source_preparation"]["codec_cache_hit"] is True
    assert calls.count("encode") == 1
    second_take = load_conditioning_take(state.job(second["id"])["source_preparation"]["take_file"])[0]
    assert second_take.text == "<|speaker:0|> Corrected words."
    assert all(job["mode"] != "audio_import" for job in state.jobs())


@pytest.mark.parametrize("seconds", [.1, 120.1])
def test_upload_duration_validated_before_model_work(importer, seconds):
    state, tasks = importer
    upload = recording(state, seconds)
    with pytest.raises(ValueError, match="0.5 to 120"):
        state.start_uploaded_audio_edit(request(upload))
    assert not tasks and not state.jobs()


def test_transcript_boundaries_and_native_support_checked_before_queue(importer, monkeypatch):
    state, tasks = importer
    upload = recording(state)
    for payload in [request(upload, original_transcript=""), request(upload, text=""),
                    request(upload, "infill", start_frame=-1), request(upload, "infill", end_frame=1000)]:
        with pytest.raises((TypeError, ValueError)):
            state.start_uploaded_audio_edit(payload)
    monkeypatch.setattr(server, "load_native_runtime", lambda _: {"protocol": 4})
    with pytest.raises(ValueError, match="does not support"):
        state.start_uploaded_audio_edit(request(upload))
    assert not tasks and not state.jobs()


def test_cancelled_upload_edit_never_loads_weights(importer):
    state, tasks = importer
    upload = recording(state)
    state._get_engine = lambda _: pytest.fail("A cancelled upload edit must not load weights")
    created = state.start_uploaded_audio_edit(request(upload))
    state.cancel(created["id"])
    run_next(tasks)
    assert state.job(created["id"])["status"] == "cancelled"
    assert state.job(created["id"])["output"] is None


def test_imported_codes_seed_python_reference_cache_without_encoding(tmp_path):
    from collections import OrderedDict
    from threading import RLock
    from drama_low_vram.engine import DramaEngine
    from drama_low_vram.gguf_takes import CODEC_CONFIG

    engine = DramaEngine.__new__(DramaEngine)
    engine._lock = RLock()
    engine._closed = False
    engine._transition_failure = None
    engine.model = SimpleNamespace(config=CODEC_CONFIG)
    engine.options = SimpleNamespace(reference_cache_entries=2)
    engine._references = OrderedDict()
    engine._run_codec = lambda *_args, **_kwargs: pytest.fail("Imported codes must not be encoded again")
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"immutable source")
    codes = torch.ones(10, 44, dtype=torch.int64)
    for text in ("<|speaker:0|> Original.", "<|speaker:0|> Corrected."):
        engine.cache_reference_codes(audio, text, codes)
        cached, hit = engine.prepare_reference(audio, text)
        assert hit and torch.equal(cached, codes)
