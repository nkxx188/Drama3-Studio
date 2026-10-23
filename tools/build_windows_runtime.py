"""Build a relocatable NVIDIA worker for RTX20/30/40/50, preserving research CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from build_native import windows_tools
from prepare_sources import ensure_checkout

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("75-real", "86-real", "89-real", "120-real", "120-virtual")
UPSTREAM_COMMIT = "624cf918e51263d39c2a0d28a6c5e3d55990b79f"


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/nvidia-universal-build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime/native")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--executable-name", default="drama3_worker.exe")
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/audio-cpp-phase19")
    parser.add_argument("--cuda-toolkit", type=Path, default=ROOT / ".cache/toolchains/cuda-12.8")
    parser.add_argument("--cmake", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Build jobs must be positive")
    receipt_file = ROOT / "research/phase35-worker-build.json"
    baseline = json.loads(receipt_file.read_text()) if receipt_file.is_file() else {}
    preserved = {name: expected for name, expected in baseline.get("existing_cli_hashes", {}).items()
                 if Path(name).is_file()}
    if baseline.get("executable") and Path(baseline["executable"]).is_file():
        preserved[baseline["executable"]] = baseline["sha256"]
    source, kit = args.source.resolve(), args.cuda_toolkit.resolve()
    if (Path(args.executable_name).name != args.executable_name
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.exe", args.executable_name)):
        raise ValueError("--executable-name must be a plain .exe filename")
    if not (kit / "bin/nvcc.exe").is_file():
        raise RuntimeError("Select a CUDA 12.8 toolkit with --cuda-toolkit; the runtime package does not need this SDK.")
    patch = ROOT / "native/audio-cpp-runtime.patch"
    metadata = json.loads((ROOT / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    if metadata["upstream"] != UPSTREAM_COMMIT or sha(patch) != metadata["sha256"]:
        raise RuntimeError("Pinned native source or patch checksum differs from its metadata")
    ensure_checkout(source, "https://github.com/0xShug0/audio.cpp.git", UPSTREAM_COMMIT, patch)
    build, output = args.build_dir.resolve(), args.output_dir.resolve()
    env, cmake, ninja, vs = windows_tools()
    bundled_cmake = ROOT / ".cache/toolchains/python/cmake/data/bin/cmake.exe"
    cmake = str(args.cmake or (bundled_cmake if bundled_cmake.is_file() else cmake))
    env = {key.upper(): value for key, value in env.items()}
    (ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"), CUDA_PATH=str(kit),
               CUDAToolkit_ROOT=str(kit), VSLANG="1033")
    env["PATH"] = str(kit / "bin") + os.pathsep + env["PATH"]
    configure = [
        cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_SYSTEM_NAME=Windows", "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
        f"-DCUDAToolkit_ROOT={kit}", f"-DCMAKE_CUDA_COMPILER={kit / 'bin/nvcc.exe'}",
        "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(ARCHITECTURES),
        "-DENGINE_ENABLE_CUDA=ON", "-DENGINE_ENABLE_VULKAN=OFF", "-DENGINE_ENABLE_HIP=OFF",
        "-DENGINE_ENABLE_METAL=OFF", "-DENGINE_ENABLE_LLAMAFILE=OFF", "-DENGINE_ENABLE_NATIVE_CPU=OFF",
        "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=OFF",
        "-DAUDIOCPP_MODEL_SET=custom", "-DAUDIOCPP_MODELS=fish_drama3",
        "-DAUDIOCPP_DEPLOYMENT_BUILD=ON", "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF",
        "-DENGINE_BUILD_TESTS=OFF",
        "-DDRAMA_CODEC_PROBE_SOURCE:FILEPATH=" + (ROOT / "native/gguf_codec_probe.cpp").as_posix(),
        "-DDRAMA_WORKER_SOURCE:FILEPATH=" + (ROOT / "native/gguf_worker.cpp").as_posix(),
    ]
    log = ROOT / "research/phase37-universal-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(configure, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([cmake, "--build", str(build), "--target", "drama3_worker",
                        "--parallel", str(args.jobs)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    output.mkdir(parents=True, exist_ok=True)
    executable = output / args.executable_name
    shutil.copy2(build / "bin/drama3_worker.exe", executable)
    # Worker is linked against shared CUDA math libraries, not the CUDA SDK.
    for pattern in ("cudart64*.dll", "cublas64*.dll", "cublasLt64*.dll", "cufft64*.dll"):
        for file in (kit / "bin").glob(pattern):
            shutil.copy2(file, output / file.name)
    license_root = output / "licenses/cuda"
    if (kit / "licenses").is_dir():
        shutil.copytree(kit / "licenses", license_root, dirs_exist_ok=True)
    for name, original in (("audio-cpp", source / "LICENSE"),
                           ("ggml", source / "external/ggml/LICENSE")):
        if original.is_file():
            target = output / "licenses" / name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target / "LICENSE")
    # Include the redistributable MSVC CRT instead of relying on the build PC.
    redist = Path(vs) / "VC/Redist/MSVC"
    crt_dirs = sorted(redist.glob("*/x64/Microsoft.VC*.CRT"))
    if crt_dirs:
        for file in crt_dirs[-1].glob("*.dll"):
            shutil.copy2(file, output / file.name)
    openmp_dirs = sorted(redist.glob("*/x64/Microsoft.VC*.OpenMP"))
    if openmp_dirs:
        for file in openmp_dirs[-1].glob("*.dll"):
            shutil.copy2(file, output / file.name)
    manifest = {
        "schema_version": 1, "executable": args.executable_name, "sha256": sha(executable),
        "cuda_toolkit": ".", "source": ".", "architectures": [75, 86, 89, 120],
        "compiled_code_targets": sorted(set(re.findall(
            r"code=\[(sm_[0-9]+a?)\]", (build / "build.ninja").read_text(encoding="utf-8")))),
        "compiled_ptx_targets": sorted(set(re.findall(
            r"code=\[(compute_[0-9]+a?)\]", (build / "build.ninja").read_text(encoding="utf-8")))),
        "platform": "windows-x86_64", "cpu_requires_avx2": True,
        "cuda_version": "12.8", "protocol": 5,
        "features": ["ordinary", "references", "previous_takes", "stream", "infill",
                     "audio_stream", "design", "joint_dialogue", "reference_prewarm"],
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.suffix in (".dll", ".exe")},
        "scope": "Compiled targets, not a claim of physical testing on every architecture",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    receipt = {"output": str(output), "build": str(build), "manifest": manifest, "command": configure}
    if not all(sha(name) == expected for name, expected in preserved.items()):
        raise RuntimeError("A previously validated research binary changed during the isolated build")
    receipt["preserved_research_binary_hashes"] = preserved
    (ROOT / "research/phase37-universal-build.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                              encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
