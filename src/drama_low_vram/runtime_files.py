"""Resolve installed runtime files without build-PC paths."""

import json
from pathlib import Path


def load_native_runtime(root):
    root = Path(root).resolve()
    folder = root / "runtime/native"
    manifest = folder / "manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        executable = (folder / value["executable"]).resolve()
        if not executable.is_relative_to(folder.resolve()) or not executable.is_file():
            raise RuntimeError("Packaged native executable is missing or outside its runtime directory")
        return value | {
            "executable": str(executable), "source": str(root),
            "cuda_toolkit": str(folder), "dll_directory": str(folder),
            "portable": True, "manifest": str(manifest),
        }
    receipt = root / "research/phase35-worker-build.json"
    if not receipt.is_file():
        raise RuntimeError("Native runtime is missing. Install the Windows runtime package or build it.")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if not Path(value["executable"]).is_file():
        raise RuntimeError("Research runtime path is unavailable; install/build the portable Windows runtime.")
    return value | {"protocol": 1, "portable": False}
