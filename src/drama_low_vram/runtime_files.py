"""Resolve installed native runtime files without build-PC paths.

A portable package may carry two manifests: ``manifest.json`` for Windows and
``manifest-linux.json`` for Linux. The platform-specific manifest wins when it
exists, so one checkout can be used on either OS without editing files.
"""

import json
import os
from pathlib import Path


def load_native_runtime(root, *, platform=None):
    root = Path(root).resolve()
    folder = root / "runtime/native"
    platform = platform or ("windows" if os.name == "nt" else "linux")
    candidates = ["manifest.json"]
    if platform == "linux":
        candidates.insert(0, "manifest-linux.json")
    for name in candidates:
        manifest = folder / name
        if not manifest.is_file():
            continue
        value = json.loads(manifest.read_text(encoding="utf-8"))
        declared = str(value.get("platform", ""))
        if (platform == "linux" and declared.startswith("windows")
                or platform == "windows" and declared.startswith("linux")):
            raise RuntimeError(
                f"Native runtime manifest {name} targets {declared}, not this platform. "
                "Build the native runtime for the current operating system."
            )
        executable = (folder / value["executable"]).resolve()
        if not executable.is_relative_to(folder.resolve()) or not executable.is_file():
            raise RuntimeError("Packaged native executable is missing or outside its runtime directory")
        if platform == "linux" and declared.startswith("linux") and not os.access(executable, os.X_OK):
            raise RuntimeError(f"Packaged native worker is not executable: {executable}")
        return value | {
            "executable": str(executable), "source": str(root),
            "cuda_toolkit": str(folder), "dll_directory": str(folder),
            "portable": True, "manifest": str(manifest),
        }
    if platform == "linux":
        raise RuntimeError(
            "Linux native runtime is missing. Build it with "
            "'python3 tools/build_linux_runtime.py' (CUDA 12.8 toolkit required)."
        )
    receipt = root / "research/phase35-worker-build.json"
    if not receipt.is_file():
        raise RuntimeError("Native runtime is missing. Install the matching runtime package or build it.")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if not Path(value["executable"]).is_file():
        raise RuntimeError("Research runtime path is unavailable; install/build the matching native runtime.")
    return value | {"protocol": 1, "portable": False}
