"""Build the relocatable NVIDIA worker for Linux x86_64 from the pinned audio.cpp source.

Same pin, same patch and same CMake flags as ``build_windows_runtime.py``; only the
toolchain plumbing differs. Output lands in ``runtime/native`` so Linux and Windows
packages stay interchangeable from the application's point of view.

Requirements: a C++17 host compiler, CMake, Ninja and a CUDA toolkit (12.8 or 12.9).
The toolkit is only needed on the build machine; the resulting worker links against
the shared CUDA math libraries shipped by the ``nvidia-*-cu12`` wheels, not the SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_sources import ensure_checkout

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "624cf918e51263d39c2a0d28a6c5e3d55990b79f"
DEFAULT_ARCHITECTURES = "89"
C23_GUARD = "#if __GLIBC_USE (IEC_60559_FUNCS_EXT_C23)"


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def cuda_version(nvcc: Path) -> str:
    report = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True)
    match = re.search(r"release (\d+\.\d+)", report.stdout)
    if match is None:
        raise RuntimeError(f"Cannot determine the CUDA version from {nvcc}")
    return match.group(1)


def locate_mathcalls() -> Path | None:
    """Return the system ``bits/mathcalls.h`` only when it enables the C23 math guard."""
    for base in ("/usr/include/x86_64-linux-gnu", "/usr/include"):
        candidate = Path(base) / "bits/mathcalls.h"
        if candidate.is_file() and C23_GUARD in candidate.read_text(errors="ignore"):
            return candidate
    return None


def c23_sysinclude(cache_root: Path) -> Path | None:
    """Neutralise the glibc C23 ``rsqrtf`` specifier clash seen by nvcc.

    glibc 2.43 declares the C23 float math functions with ``noexcept(true)``. nvcc
    re-declares the same names as ``__device__`` functions, so device compilation
    aborts with an exception-specification mismatch. Overriding one shared header in
    a private include directory keeps the system headers untouched and is a no-op on
    older glibc, where the guard does not exist.
    """
    source = locate_mathcalls()
    if source is None:
        return None
    target = cache_root / "sysinclude/bits/mathcalls.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text().replace(
        C23_GUARD, "#if 0 /* patched: nvcc declares these as device functions */", 1))
    print(f"glibc C23 math guard neutralised via {target}", flush=True)
    return target.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/linux-worker-build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime/native")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/audio-cpp")
    parser.add_argument("--cuda-toolkit", type=Path,
                        help="CUDA toolkit root; defaults to $CUDA_HOME or /usr/local/cuda")
    parser.add_argument("--architectures", default=DEFAULT_ARCHITECTURES,
                        help="CMake CUDA architectures, e.g. '89' or '75;86;89;120'")
    parser.add_argument("--cmake", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Build jobs must be positive")
    if sys.platform != "linux":
        raise RuntimeError("This script targets Linux; use build_windows_runtime.py on Windows")

    kit = Path(args.cuda_toolkit or os.environ.get("CUDA_HOME") or "/usr/local/cuda").resolve()
    nvcc = kit / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA toolkit not found at {kit}; pass --cuda-toolkit")
    patch = ROOT / "native/audio-cpp-runtime.patch"
    metadata = json.loads((ROOT / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    if metadata["upstream"] != UPSTREAM_COMMIT or sha(patch) != metadata["sha256"]:
        raise RuntimeError("Pinned native source or patch checksum differs from its metadata")

    source = args.source.resolve()
    ensure_checkout(source, "https://github.com/0xShug0/audio.cpp.git", UPSTREAM_COMMIT, patch)
    build, output = args.build_dir.resolve(), args.output_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)

    cuda_flags = []
    sysinclude = c23_sysinclude(ROOT / ".cache")
    if sysinclude is not None:
        cuda_flags.append("-I" + sysinclude.as_posix())

    env = dict(os.environ)
    env["PATH"] = str(kit / "bin") + os.pathsep + env.get("PATH", "")
    env["CUDAToolkit_ROOT"] = str(kit)
    # nvcc rejects host compilers newer than its own support matrix; the worker only
    # needs C++17, so allow the newer host compiler instead of pinning an old one.
    env.setdefault("NVCC_PREPEND_FLAGS", "-allow-unsupported-compiler")

    configure = [
        str(args.cmake or shutil.which("cmake") or "cmake"),
        "-S", str(source), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCUDAToolkit_ROOT={kit}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
        "-DCMAKE_CUDA_FLAGS=" + " ".join(cuda_flags),
        "-DENGINE_ENABLE_CUDA=ON", "-DENGINE_ENABLE_VULKAN=OFF", "-DENGINE_ENABLE_HIP=OFF",
        "-DENGINE_ENABLE_METAL=OFF", "-DENGINE_ENABLE_LLAMAFILE=OFF", "-DENGINE_ENABLE_NATIVE_CPU=OFF",
        "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=OFF",
        "-DAUDIOCPP_MODEL_SET=custom", "-DAUDIOCPP_MODELS=fish_drama3",
        "-DAUDIOCPP_DEPLOYMENT_BUILD=ON", "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF",
        "-DENGINE_BUILD_TESTS=OFF",
        "-DDRAMA_CODEC_PROBE_SOURCE:FILEPATH=" + (ROOT / "native/gguf_codec_probe.cpp").as_posix(),
        "-DDRAMA_WORKER_SOURCE:FILEPATH=" + (ROOT / "native/gguf_worker.cpp").as_posix(),
    ]
    log = build / "build.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(configure, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([configure[0], "--build", str(build), "--target", "drama3_worker",
                        "--parallel", str(args.jobs)], env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

    output.mkdir(parents=True, exist_ok=True)
    executable = output / "drama3_worker"
    shutil.copy2(build / "bin/drama3_worker", executable)
    executable.chmod(0o755)
    for name, original in (("audio-cpp", source / "LICENSE"),
                           ("ggml", source / "external/ggml/LICENSE")):
        if original.is_file():
            target = output / "licenses" / name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target / "LICENSE")

    ninja_file = (build / "build.ninja").read_text(encoding="utf-8")
    manifest = {
        "schema_version": 1, "platform": "linux-x86_64", "executable": executable.name,
        "sha256": sha(executable),
        "cuda_toolkit": ".", "source": ".",
        "architectures": [int(item) for item in re.findall(r"\d+", args.architectures)],
        "compiled_code_targets": sorted(set(re.findall(r"code=\[(sm_[0-9]+a?)\]", ninja_file))),
        "compiled_ptx_targets": sorted(set(re.findall(r"code=\[(compute_[0-9]+a?)\]", ninja_file))),
        "cpu_requires_avx2": True,
        "cuda_version": cuda_version(nvcc),
        "protocol": 5,
        "features": ["ordinary", "references", "previous_takes", "stream", "infill",
                     "audio_stream", "design", "joint_dialogue", "reference_prewarm",
                     "output_codec_prewarm"],
        "scope": "Compiled targets, not a claim of physical testing on every architecture",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
