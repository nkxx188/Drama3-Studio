"""Resolve installed native runtime files without build-PC paths.

A checkout can have a Windows runtime in ``runtime/native`` and a separate
Linux runtime in ``runtime/native-linux``. Docker stages its own runtime under
``runtime/native`` inside the container.
"""

import json
import os
from pathlib import Path


def find_native_manifest(root, *, platform=None):
    platform = platform or ("windows" if os.name == "nt" else "linux")
    runtime = Path(root) / "runtime"
    candidates = [runtime / "native/manifest.json"]
    if platform == "linux":
        candidates[:0] = [
            runtime / "native-linux/manifest-linux.json",
            runtime / "native/manifest-linux.json",
        ]
    return next((manifest for manifest in candidates if manifest.is_file()), None)


def load_native_runtime(root, *, platform=None):
    root = Path(root).resolve()
    platform = platform or ("windows" if os.name == "nt" else "linux")
    manifest = find_native_manifest(root, platform=platform)
    if manifest is not None:
        folder = manifest.parent
        value = json.loads(manifest.read_text(encoding="utf-8"))
        declared = str(value.get("platform", ""))
        if (platform == "linux" and not declared.startswith("linux")
                or platform == "windows" and declared.startswith("linux")):
            raise RuntimeError(
                f"Native runtime manifest {manifest.name} targets {declared or 'an unknown OS'}, not this platform. "
                "Build the native runtime for the current operating system."
            )
        executable = (folder / value["executable"]).resolve()
        if not executable.is_relative_to(folder.resolve()) or not executable.is_file():
            raise RuntimeError("Packaged native executable is missing or outside its runtime directory")
        if platform == "linux" and not os.access(executable, os.X_OK):
            raise RuntimeError(f"Packaged native worker is not executable: {executable}")
        return value | {
            "executable": str(executable), "source": str(root),
            "cuda_toolkit": str(folder), "dll_directory": str(folder),
            "portable": True, "manifest": str(manifest),
        }
    if platform == "linux":
        raise RuntimeError(
            "Linux native runtime is missing. Build it with "
            "'.venv-linux/bin/python tools/build_linux_runtime.py'. See docs/linux.md."
        )
    receipt = root / "research/phase35-worker-build.json"
    if not receipt.is_file():
        raise RuntimeError("Native runtime is missing. Install the Windows runtime package or build it.")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if not Path(value["executable"]).is_file():
        raise RuntimeError("Research runtime path is unavailable; install/build the portable Windows runtime.")
    return value | {"protocol": 1, "portable": False}
