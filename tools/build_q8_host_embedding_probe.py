"""Link the optional host-lookup probe and restore the original native CLI."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-sweep", type=Path)
    args = parser.parse_args()
    if args.wait_sweep:
        started = time.perf_counter()
        while json.loads(args.wait_sweep.read_text())["status"] == "running":
            if time.perf_counter() - started > 1200:
                raise TimeoutError("Sweep did not finish; original CLI remains untouched")
            time.sleep(1)
    build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    original = Path(build["executable"])
    backup = ROOT / ".cache/phase29-native-before-host-embedding"
    if sha(backup / "audiocpp_cli.exe") != build["sha256"] or sha(original) != build["sha256"]:
        raise ValueError("Original CLI/backup identity changed; stop before linking")
    source = Path(build["source"])
    files = [
        "src/community_models/fish_drama3/backbone.cpp",
        "src/framework/core/disk_expert_weight_store.cpp",
        "include/engine/framework/core/disk_expert_weight_store.h",
    ]
    patches = []
    for name in files:
        before = (backup / Path(name).name).read_text()
        after = (source / name).read_text()
        patches.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name))
    patch = ROOT / "research/phase29-audio-cpp-host-embedding.patch"
    patch.write_text("".join(patches), encoding="utf-8")
    env, _, _, _ = windows_tools()
    env = {key.upper(): value for key, value in env.items()}
    kit = Path(build["cuda_toolkit"])
    env.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"),
               CUDA_PATH=str(kit), CUDAToolkit_ROOT=str(kit), VSLANG="1033")
    env["PATH"] = str(kit / "bin") + os.pathsep + env["PATH"]
    probe = original.with_name("audiocpp_cli-host-embedding.exe")
    log_path = ROOT / "research/phase29-host-embedding-link.log"
    print("Linking isolated host-embedding probe", flush=True)
    try:
        with log_path.open("w") as output:
            linked = subprocess.run([
                build["cmake"], "--build", str(original.parent.parent),
                "--target", "audiocpp_cli", "--parallel", "2",
            ], env=env, stdout=output, stderr=subprocess.STDOUT)
        if linked.returncode:
            raise RuntimeError(f"Probe link failed; see {log_path}")
        shutil.copy2(original, probe)
    finally:
        shutil.copy2(backup / "audiocpp_cli.exe", original)
        if sha(original) != build["sha256"]:
            raise ValueError("Original CLI restore verification failed")
    receipt = build | {
        "executable": str(probe), "sha256": sha(probe),
        "scope": "Opt-in mapped BF16 host-token lookup probe; original CLI restored byte-for-byte.",
        "probe_environment": {"DRAMA_HOST_GGUF_EMBEDDING": "1"},
        "patch": str(patch), "patch_sha256": sha(patch),
        "modified_sources": {name: sha(source / name) for name in files},
        "original_cli_sha256_after_restore": sha(original),
    }
    destination = ROOT / "research/phase29-host-embedding-build.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
