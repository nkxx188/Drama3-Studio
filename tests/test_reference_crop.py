"""A waveform crop creates exact source samples without modifying the original."""

import io
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import ui.server as server


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    samples = np.stack((
        np.sin(np.arange(48000) * 0.03),
        np.cos(np.arange(48000) * 0.04),
    ), axis=1).astype("float32") * 0.5
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="FLOAT")
    uploaded = state.add_reference_upload("reference.wav", buffer.getvalue())
    try:
        yield state, uploaded, samples, buffer.getvalue()
    finally:
        state.shutdown()


def test_crop_preserves_exact_samples_channels_rate_and_original(reference):
    state, upload, samples, original = reference
    result = state.crop_reference_upload({"id": upload["id"], "start": 0.5001, "end": 1.6001})
    assert result["id"] != upload["id"]
    target = state.uploads_root / result["id"] / "reference.wav"
    actual, rate = sf.read(target, dtype="float32", always_2d=True)
    first, last = math.floor(0.5001 * rate), math.ceil(1.6001 * rate)
    assert rate == 16000 and np.array_equal(actual, samples[first:last])
    assert result["samples"] == last - first
    assert result["audio_url"].startswith("/files/ui_data/uploads/")
    source = state.uploads_root / upload["id"] / "reference.wav"
    assert source.read_bytes() == original


@pytest.mark.parametrize("start,end", [
    (-1, 1), (2, 1), (1, 1), (0, 0.1), (0, 4), (True, 2),
    (float("nan"), 2), (0, float("inf")),
])
def test_invalid_crop_never_creates_a_partial_reference(reference, start, end):
    state, upload, _, _ = reference
    with pytest.raises(ValueError):
        state.crop_reference_upload({"id": upload["id"], "start": start, "end": end})
    assert len(list(state.uploads_root.iterdir())) == 1


def test_crop_rejects_path_escape_before_opening_files(reference):
    state, _, _, _ = reference
    with pytest.raises(ValueError, match="upload ID"):
        state.crop_reference_upload({"id": "../voices", "start": 0, "end": 1})
