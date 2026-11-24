"""Build a relocatable Linux NVIDIA worker with the same pinned recipe as Windows.

The worker is compiled from the pinned audio.cpp checkout with the exact same
CUDA target list and feature flags as ``build_windows_runtime.py``. Only
the platform packaging differs: an ELF executable plus the CUDA shared objects
it needs are staged in ``runtime/native-linux`` and described by ``manifest-linux.json``.
No system-wide CUDA installation is needed at run time (the driver is enough).

Requires: CMake, Ninja, a C++17 compiler, and CUDA 13.x (nvcc).
On older distributions CUDA 12.8 also works.

Usage:
    python3 tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_sources import ensure_checkout

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("75-real", "86-real", "89-real", "120-real", "120-virtual")
UPSTREAM_COMMIT = "624cf918e51263d39c2a0d28a6c5e3d55990b79f"
EXECUTABLE = "drama3_worker"
MANIFEST_NAME = "manifest-linux.json"
PREFIX = "@drama3 "
CUDA_LIB_PATTERNS = (
    "libcudart.so*", "libcublas.so*", "libcublasLt.so*", "libcufft.so*",
    "libcurand.so*", "libcusparse.so*", "libnvrtc.so*", "libnpp*.so*",
    "libnvJitLink.so*", "libnvfatbin.so*",
)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_patch():
    metadata = json.loads((ROOT / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    if metadata["upstream"] != UPSTREAM_COMMIT:
        raise RuntimeError("Pinned native source commit differs from the build recipe")
    # The metadata was authored on Windows and stores the path with a backslash.
    relative = metadata["patch"].replace("\\", "/")
    patch = (ROOT / relative).resolve()
    if not patch.is_file() or sha(patch) != metadata["sha256"]:
        raise RuntimeError("Pinned native source patch differs from its metadata")
    return patch


def toolkit_dirs(kit):
    # Never copy linker stubs: they contain no working device code.
    return sorted({kit / "lib64", kit / "lib"})


def default_host_compiler():
    """Pick a conservative host compiler (GCC <= 14 when available)."""
    gpp = shutil.which("g++")
    if gpp:
        try:
            version = subprocess.run([gpp, "-dumpversion"], capture_output=True,
                                     text=True, check=True).stdout.strip()
            if int(version.split(".")[0]) <= 14:
                return gpp
        except (ValueError, OSError, subprocess.CalledProcessError):
            pass
    for candidate in ("/usr/bin/g++-14", "/usr/bin/g++-13", "/usr/bin/g++-12"):
        if Path(candidate).is_file():
            return candidate
    return None


def toolkit_version(nvcc):
    result = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=True)
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout + result.stderr)
    if not match:
        raise RuntimeError(f"Could not determine the CUDA toolkit version from {nvcc}")
    version = (int(match.group(1)), int(match.group(2)))
    if version < (12, 8):
        raise RuntimeError("CUDA 12.8 or newer is required for the sm_120 target")
    return f"{version[0]}.{version[1]}"


def find_libraries(kit, worker):
    """Collect the bundled CUDA objects the worker actually loads."""
    found = {}
    process = subprocess.run(["ldd", str(worker)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    for line in process.stdout.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+?)\s+\(", line)
        if not match:
            continue
        link = Path(match.group(2))
        resolved = link.resolve()
        # Keep the link name: it is the SONAME the loader asks for at run time.
        if resolved.is_file() and resolved.is_relative_to(kit):
            found[link.name] = resolved
    for folder in toolkit_dirs(kit):
        if not folder.is_dir():
            continue
        for pattern in CUDA_LIB_PATTERNS:
            for link in folder.glob(pattern):
                resolved = link.resolve()
                if resolved.is_file() and resolved.is_relative_to(kit):
                    found.setdefault(link.name, resolved)
    return found


def set_rpath(executable):
    """Make the worker find bundled libraries even without LD_LIBRARY_PATH."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        return False
    subprocess.run([patchelf, "--set-rpath", "$ORIGIN", str(executable)], check=True)
    return True


def probe(worker, output):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(output) + (
        os.pathsep + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    environment["PATH"] = str(output) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run([str(worker), "--probe"], cwd=str(output), env=environment,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
    rows = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()
            if line.startswith(PREFIX)]
    return result.returncode, rows, result.stderr[-2000:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/linux-nvidia-build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime/native-linux")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/audio-cpp-linux")
    parser.add_argument("--cuda-toolkit", type=Path,
                        default=Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")))
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    parser.add_argument("--cuda-host-compiler", default=None,
                        help="Host C++ compiler for nvcc (auto-detected by default)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Build jobs must be positive")
    if sys.platform != "linux":
        parser.error("This script builds the Linux runtime. Use build_windows_runtime.py on Windows.")

    kit = args.cuda_toolkit.resolve()
    nvcc = kit / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA toolkit with bin/nvcc is required: {kit}. "
                           "Pass --cuda-toolkit or set CUDA_PATH.")
    cuda_version = toolkit_version(nvcc)
    cmake, ninja = shutil.which(args.cmake), shutil.which(args.ninja)
    if not cmake or not ninja:
        raise RuntimeError("CMake and Ninja are required to build the native worker")

    patch = verify_patch()
    source, build = args.source.resolve(), args.build_dir.resolve()
    ensure_checkout(source, "https://github.com/0xShug0/audio.cpp.git", UPSTREAM_COMMIT, patch)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    (ROOT / ".cache/tmp").mkdir(parents=True, exist_ok=True)
    environment.update(TMPDIR=str(ROOT / ".cache/tmp"), TEMP=str(ROOT / ".cache/tmp"),
                       TMP=str(ROOT / ".cache/tmp"), CUDA_PATH=str(kit),
                       CUDAToolkit_ROOT=str(kit))
    environment["PATH"] = str(kit / "bin") + os.pathsep + environment.get("PATH", "")
    host_compiler = args.cuda_host_compiler or default_host_compiler()
    if host_compiler:
        environment["CXX"] = host_compiler
        environment["CC"] = host_compiler.replace("g++", "gcc")
    configure = [
        cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCUDAToolkit_ROOT={kit}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
        *([f"-DCMAKE_CUDA_HOST_COMPILER={host_compiler}"] if host_compiler else []),
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
    log = ROOT / "research/phase37-linux-build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(configure, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([cmake, "--build", str(build), "--target", EXECUTABLE,
                        "--parallel", str(args.jobs)], env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

    built = build / "bin" / EXECUTABLE
    if not built.is_file():
        raise RuntimeError(f"Expected worker binary missing: {built}. See {log}")
    executable = output / EXECUTABLE
    shutil.copy2(built, executable)
    executable.chmod(0o755)
    patchelf_used = set_rpath(executable)

    libraries = find_libraries(kit, built)
    for name, path in sorted(libraries.items()):
        shutil.copy2(path, output / name)

    for name, original in (("audio-cpp", source / "LICENSE"),
                           ("ggml", source / "external/ggml/LICENSE")):
        if original.is_file():
            target = output / "licenses" / name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target / "LICENSE")
    if (kit / "LICENSE").is_file():
        target = output / "licenses/cuda"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kit / "LICENSE", target / "LICENSE")

    code = (build / "build.ninja").read_text(encoding="utf-8")
    manifest = {
        "schema_version": 1, "executable": EXECUTABLE, "sha256": sha(executable),
        "cuda_toolkit": ".", "source": ".", "architectures": [75, 86, 89, 120],
        "compiled_code_targets": sorted(set(re.findall(r"code=\[(sm_[0-9]+a?)\]", code))),
        "compiled_ptx_targets": sorted(set(re.findall(r"code=\[(compute_[0-9]+a?)\]", code))),
        "platform": "linux-x86_64", "cpu_requires_avx2": True,
        "cuda_version": cuda_version, "protocol": 5,
        "features": ["ordinary", "references", "previous_takes", "stream", "infill",
                     "audio_stream", "design", "joint_dialogue", "reference_prewarm",
                     "output_codec_prewarm"],
        "files": {path.name: sha(path) for path in output.iterdir()
                  if path.is_file() and (path.suffix == ".so" or ".so." in path.name or path == executable)},
        "scope": "Compiled targets, not a claim of physical testing on every architecture",
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    receipt = {"output": str(output), "build": str(build), "manifest": manifest,
               "cuda_toolkit": str(kit), "cuda_version": cuda_version,
               "command": configure, "patchelf_rpath": patchelf_used,
               "bundled_libraries": sorted(libraries)}
    (ROOT / "research/phase37-linux-build.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    returncode, rows, errors = probe(executable, output)
    if returncode or not rows or not rows[-1].get("available"):
        detail = rows[-1].get("error", "") if rows else errors
        raise RuntimeError("Native worker was built but its GPU probe failed. "
                           "Check the NVIDIA driver and the bundled CUDA libraries. " + detail)
    gpu_names = ", ".join(device["name"] for device in rows[-1].get("devices", []))
    print(json.dumps({"manifest": MANIFEST_NAME, "executable": str(executable),
                      "bundled_libraries": sorted(libraries), "patchelf_rpath": patchelf_used,
                      "probe": rows[-1], "gpus": gpu_names}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Native worker build failed (exit {error.returncode}). "
              f"See research/phase37-linux-build.log", file=sys.stderr)
        raise SystemExit(1) from error
