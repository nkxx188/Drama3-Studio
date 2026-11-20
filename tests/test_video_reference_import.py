"""Video containers are reduced to their audio track before editing starts."""

import io
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

import ui.server as server
from drama_low_vram import audio_import


def wav_bytes(seconds=.5, rate=16000):
    samples = np.zeros(int(seconds * rate), dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _: {"protocol": 5})
    value = server.AppState()
    try:
        yield value
    finally:
        value.shutdown()


def test_video_containers_are_detected_and_converted_to_wav_bytes(monkeypatch):
    extracted = []

    def extract(data, suffix=".mp4", root=None):
        extracted.append((data, suffix))
        return b"wav audio"

    monkeypatch.setattr(audio_import, "extract_video_audio", extract)
    assert audio_import.prepare_reference_audio(b"movie", "scene.MP4") == (b"wav audio", ".wav")
    assert extracted == [(b"movie", ".mp4")]
    assert audio_import.prepare_reference_audio(b"mp3 data", "voice.mp3") == (b"mp3 data", ".mp3")
    with pytest.raises(ValueError, match="MP4"):
        audio_import.prepare_reference_audio(b"words", "notes.txt")


def test_uploaded_video_is_stored_as_the_extracted_wav(state, monkeypatch):
    audio = wav_bytes()
    monkeypatch.setattr(server, "prepare_reference_audio", lambda data, filename, root: (audio, ".wav"))
    upload = state.add_reference_upload("scene.mp4", b"\0" * 64 + b"video payload")
    stored = state.reference_upload(upload["id"])
    assert stored["filename"] == "scene.mp4"
    assert Path(stored["audio_path"]).name == "reference.wav"
    assert Path(stored["audio_path"]).read_bytes() == audio


def test_unknown_container_is_rejected_before_storage(state):
    with pytest.raises(ValueError, match="MP4"):
        state.add_reference_upload("notes.txt", b"\0" * 64)


@pytest.mark.skipif(audio_import.ffmpeg_path() is None, reason="FFmpeg is required to build the sample video")
def test_ffmpeg_keeps_only_the_audio_track(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run([
        audio_import.ffmpeg_path(), "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(clip),
    ], check=True, capture_output=True)
    audio, suffix = audio_import.prepare_reference_audio(clip.read_bytes(), clip.name)
    info = sf.info(io.BytesIO(audio))
    assert suffix == ".wav" and 1.5 < info.duration < 2.6 and info.channels == 1

    silent = tmp_path / "silent.mp4"
    subprocess.run([
        audio_import.ffmpeg_path(), "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10", "-an", str(silent),
    ], check=True, capture_output=True)
    with pytest.raises(ValueError, match="audio track"):
        audio_import.prepare_reference_audio(silent.read_bytes(), silent.name)
