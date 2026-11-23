"""Portable runtime selection tests shared by Windows and Linux checkouts."""

import json
from pathlib import Path

import pytest

from drama_low_vram.runtime_files import load_native_runtime


def _native(root, name, platform, *, executable="drama3_worker", mode=0o755):
    folder = root / "runtime/native"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / executable
    target.write_bytes(b"worker")
    target.chmod(mode)
    (folder / name).write_text(json.dumps({
        "executable": executable, "platform": platform, "protocol": 5,
    }), encoding="utf-8")
    return folder


def test_linux_manifest_is_preferred_on_linux(tmp_path):
    folder = _native(tmp_path, "manifest.json", "windows-x86_64",
                     executable="drama3_worker.exe")
    _native(tmp_path, "manifest-linux.json", "linux-x86_64")
    result = load_native_runtime(tmp_path, platform="linux")
    assert result["executable"] == str(folder / "drama3_worker")
    assert result["dll_directory"] == str(folder)
    assert result["protocol"] == 5


def test_windows_manifest_still_used_for_windows(tmp_path):
    folder = _native(tmp_path, "manifest.json", "windows-x86_64",
                     executable="drama3_worker.exe")
    _native(tmp_path, "manifest-linux.json", "linux-x86_64")
    result = load_native_runtime(tmp_path, platform="windows")
    assert result["executable"] == str(folder / "drama3_worker.exe")


def test_wrong_platform_manifest_is_rejected(tmp_path):
    _native(tmp_path, "manifest.json", "windows-x86_64", executable="drama3_worker.exe")
    with pytest.raises(RuntimeError, match="windows"):
        load_native_runtime(tmp_path, platform="linux")


def test_missing_linux_runtime_has_actionable_message(tmp_path):
    with pytest.raises(RuntimeError, match="build_linux_runtime.py"):
        load_native_runtime(tmp_path, platform="linux")


def test_pinned_native_patch_metadata_resolves_on_both_filesystems():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    # The metadata path was authored on Windows; Linux must normalize it.
    relative = metadata["patch"].replace("\\", "/")
    assert (root / relative).is_file()
