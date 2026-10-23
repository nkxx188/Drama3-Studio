"""Diagnostics export remains bounded and excludes model/audio/request data."""

import json
import zipfile

from drama_low_vram import diagnostics
from unittest.mock import Mock
import torch


def test_export_contains_snapshot_and_bounded_logs_not_user_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "snapshot", lambda _: {"test": "hardware"})
    folder = tmp_path / ".cache/gguf-sessions/session"
    folder.mkdir(parents=True)
    (folder / "native.log").write_text("[TIMING] test 1\n" * 50000)
    (folder / "config.json").write_text('{"secret":"request configuration"}')
    (folder / "request.wav").write_bytes(b"user-audio")
    output = diagnostics.export(tmp_path, tmp_path / "diagnostics/support.zip")
    with zipfile.ZipFile(output) as archive:
        assert json.loads(archive.read("diagnostics.json")) == {"test": "hardware"}
        assert "sessions/session/native.log" in archive.namelist()
        assert archive.getinfo("sessions/session/native.log").file_size <= 256 * 1024
        assert not any("config" in name or name.endswith(".wav") for name in archive.namelist())


def test_python_auto_device_uses_supported_gpu_with_most_free_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: [(6, 1), (7, 5), (12, 0)][i])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: [(24, 24), (8, 11), (12, 16)][i])
    assert diagnostics.select_python_device() == "cuda:2"
