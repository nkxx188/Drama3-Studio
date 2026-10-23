"""Build the isolated codec probe/CLI and preserve the original production CLI."""

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
    original_build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    original = Path(original_build["executable"])
    saved = ROOT / ".cache/phase29-native-before-host-embedding/audiocpp_cli.exe"
    if sha(original) != original_build["sha256"] or sha(saved) != original_build["sha256"]:
        raise RuntimeError("Native CLI identity changed; do not overwrite")
    env, _, _, _ = windows_tools()
    env = {key.upper(): value for key, value in env.items()}
    kit = Path(original_build["cuda_toolkit"])
    env.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"),
               CUDA_PATH=str(kit), CUDAToolkit_ROOT=str(kit), VSLANG="1033")
    env["PATH"] = str(kit / "bin") + os.pathsep + env["PATH"]
    log_path = ROOT / "research/phase30-codec-native-build.log"
    probe = original.with_name("audiocpp_cli-codec-probe.exe")
    source = Path(original_build["source"])
    backup = ROOT / ".cache/phase30-native-before-codec"
    patches = []
    relatives = [
        "CMakeLists.txt", "src/community_models/fish_drama3/generator.cpp",
        "include/engine/community_models/fish_drama3/generator.h",
        "src/community_models/fish_drama3/loader.cpp",
        "src/framework/codecs/fish_dac_codec_runtime.cpp",
    ]
    for relative in relatives:
        current = (source / relative).read_text()
        saved_source = backup / Path(relative).name
        if saved_source.exists():
            before = saved_source.read_text()
        elif relative.endswith("/loader.cpp"):
            before = current.replace(
                '"fish_drama3.codec_encoder_f32_accumulation",\n'
                '                                "fish_drama3.codec_decode_strategy", "fish_drama3.codec_decoder_backend"',
                '"fish_drama3.codec_encoder_f32_accumulation"')
            if current == before:
                raise RuntimeError("Cannot recover loader patch baseline")
            saved_source.write_text(before, encoding="utf-8")
        else:
            raise FileNotFoundError(saved_source)
        patches.extend(difflib.unified_diff(
            before.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile="a/" + relative, tofile="b/" + relative))
    patch = ROOT / "research/phase30-audio-cpp-codec.patch"
    patch.write_text("".join(patches), encoding="utf-8")
    # Old objects were produced before header dependencies were repaired.
    # Rebuild every translation unit that can allocate the changed generator.
    for path in (source / "src/community_models/fish_drama3").glob("*.cpp"):
        path.touch()
    (source / "src/framework/codecs/fish_dac_codec_runtime.cpp").touch()
    try:
        with log_path.open("w", encoding="utf-8") as output:
            configured = subprocess.run([
                original_build["cmake"], "-S", str(source), "-B", str(original.parent.parent),
                "-DDRAMA_CODEC_PROBE_SOURCE:FILEPATH=" + (ROOT / "native/gguf_codec_probe.cpp").as_posix(),
            ], env=env, stdout=output, stderr=subprocess.STDOUT)
            if configured.returncode:
                raise RuntimeError(f"Codec probe configuration failed; see {log_path}")
            result = subprocess.run([
                original_build["cmake"], "--build", str(original.parent.parent),
                "--target", "audiocpp_cli", "drama3_codec_probe", "--parallel", "2",
            ], env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Codec CLI build failed; see {log_path}")
        shutil.copy2(original, probe)
    finally:
        shutil.copy2(saved, original)
        if sha(original) != original_build["sha256"]:
            raise RuntimeError("Original native CLI restore mismatch")
    receipt = original_build | {
        "executable": str(probe), "sha256": sha(probe),
        "scope": "Isolated optional codec decoder backend/stream strategy plus phase29 host BF16 lookup.",
        "patch": str(patch), "patch_sha256": sha(patch),
        "requires_patch": str(ROOT / "research/phase29-audio-cpp-host-embedding.patch"),
        "modified_sources": {p: sha(source / p) for p in relatives},
        "codec_probe_source": str(ROOT / "native/gguf_codec_probe.cpp"),
        "codec_probe_source_sha256": sha(ROOT / "native/gguf_codec_probe.cpp"),
        "codec_probe_executable": str(original.with_name("drama3_codec_probe.exe")),
        "original_cli_sha256_after_restore": sha(original),
    }
    ninja = next(line.split("=", 1)[1] for line in
                 (original.parent.parent / "CMakeCache.txt").read_text().splitlines()
                 if line.startswith("CMAKE_MAKE_PROGRAM:FILEPATH="))
    dependencies = subprocess.check_output([
        ninja, "-C", str(original.parent.parent), "-t", "deps",
        "CMakeFiles/engine_model_fish_drama3.dir/src/community_models/fish_drama3/session.cpp.obj",
    ], env=env).decode("utf-8", errors="replace")
    if "generator.h" not in dependencies:
        raise RuntimeError("Session compiler dependencies still omit the generator header")
    dep_path = ROOT / "research/phase30-native-header-dependencies.txt"
    dep_path.write_text(dependencies, encoding="utf-8")
    receipt["session_header_dependencies_verified"] = str(dep_path)
    (ROOT / "research/phase30-codec-build.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
