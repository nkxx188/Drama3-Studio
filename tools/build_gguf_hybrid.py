"""Build the opt-in native Q8 cooperative decoder; preserve released CLIs."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((ROOT / "research/phase30-codec-build.json").read_text(encoding="utf-8"))
    source = Path(previous["source"])
    original = Path(previous["executable"]).with_name("audiocpp_cli.exe")
    backup = ROOT / ".cache/phase33-before-hybrid"
    saved = backup / original.name
    if sha(original) != sha(saved):
        raise RuntimeError("CLI changed since backup; refusing overwrite")
    env, _, _, _ = windows_tools()
    env = {key.upper(): value for key, value in env.items()}
    env["TEMP"] = env["TMP"] = str(ROOT / ".cache/tmp")
    env["PATH"] = str(Path(previous["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    relatives = [
        "CMakeLists.txt",
        "src/community_models/fish_drama3/backbone.cpp",
        "src/community_models/fish_drama3/hybrid_decode.inc",
        "include/engine/community_models/fish_drama3/backbone.h",
        "src/community_models/fish_drama3/generator.cpp",
        "src/community_models/fish_drama3/loader.cpp",
        "src/framework/core/disk_expert_weight_store.cpp",
        "include/engine/framework/core/disk_expert_weight_store.h",
        "tests/unittests/test_disk_expert_weight_store.cpp",
    ]
    patches = []
    for relative in relatives:
        before = backup / relative
        baseline = before.read_text(encoding="utf-8") if before.exists() else ""
        patches.append(f"diff --git a/{relative} b/{relative}\n")
        if not before.exists():
            patches.append("new file mode 100644\n")
        patches.extend(difflib.unified_diff(
            baseline.splitlines(True), (source / relative).read_text(encoding="utf-8").splitlines(True),
            fromfile="a/" + relative if before.exists() else "/dev/null",
            tofile="b/" + relative,
        ))
    out = ROOT / "research/phase33-native-hybrid"
    patch = out / "native-hybrid.patch"
    # LF patch text also applies to newly added LF files on Windows, while
    # Git normalizes existing CRLF worktree files when checking the hunks.
    patch.write_bytes("".join(patches).encode("utf-8"))
    # Header dependency repair is already in the local CMake opt-in block.
    # Touch the family once so old compiler objects cannot retain config layouts.
    for path in (source / "src/community_models/fish_drama3").glob("*.cpp"):
        path.touch()
    executable = original.with_name("audiocpp_cli-hybrid.exe")
    try:
        with (out / "build.log").open("wb") as log:
            subprocess.run([
                previous["cmake"], "--build", str(original.parent.parent),
                "--target", "audiocpp_cli", "--parallel", "2",
            ], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        shutil.copy2(original, executable)
    finally:
        shutil.copy2(saved, original)
    if sha(original) != sha(saved):
        raise RuntimeError("Production CLI restore failed")
    report = {
        **previous, "executable": str(executable), "sha256": sha(executable),
        "scope": "Optional Q8 cooperative single-frame decoder, unchanged batched prefill and codec.",
        "patch": str(patch), "patch_sha256": sha(patch),
        "requires_patch": str(ROOT / "research/phase32-q8-real-routing/capture-v2/route-capture.patch"),
        "modified_sources": {relative: sha(source / relative) for relative in relatives},
        "original_cli_restored_sha256": sha(original),
        "codec_cli_unchanged_sha256": sha(Path(previous["executable"])),
    }
    (ROOT / "research/phase33-hybrid-build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("native cooperative CLI built", executable, flush=True)


if __name__ == "__main__":
    main()
