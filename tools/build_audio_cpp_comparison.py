"""Build the isolated Drama3 GGUF comparison runtime using a D-drive CUDA kit."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_native import windows_tools


def component(item):
    name, data, destination = item
    archive_dir = ROOT / ".cache/toolchains/downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / Path(data["relative_path"]).name
    if not archive.is_file() or archive.stat().st_size != int(data["size"]):
        partial = archive.with_suffix(".partial")
        for attempt in range(5):
            try:
                with urlopen(Request("https://developer.download.nvidia.com/compute/cuda/redist/"
                                     + data["relative_path"], headers={"User-Agent": "DramaEngine"}),
                             timeout=60) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output, 4 * 1024**2)
                break
            except OSError:
                if attempt == 4:
                    raise
                print(json.dumps({"phase": "cuda-component-retry", "component": name,
                                  "attempt": attempt + 1}), flush=True)
                time.sleep(1)
        with partial.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != data["sha256"]:
                raise ValueError(f"CUDA component SHA256 differs: {name}")
        partial.replace(archive)
    with archive.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == data["sha256"]
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            relative = Path(*Path(member.filename).parts[1:])
            if not relative.parts:
                continue
            if relative.name.startswith(("LICENSE", "EULA")):
                relative = Path("licenses") / name / relative
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination):
                raise ValueError("CUDA archive path escaped isolated toolchain")
            if path.is_file() and path.stat().st_size == member.file_size:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as handle, path.open("wb") as output:
                shutil.copyfileobj(handle, output, 4 * 1024**2)
    print(json.dumps({"phase": "cuda-component-ready", "component": name}), flush=True)


def main():
    source = ROOT / ".cache/audio-cpp-phase19"
    if not (source / "CMakeLists.txt").is_file():
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                        "https://github.com/0xShug0/audio.cpp.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout",
                        "624cf918e51263d39c2a0d28a6c5e3d55990b79f"], check=True)
    patch = ROOT / "research/phase19-audio-cpp-local.patch"
    if patch.is_file():
        applied = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                                 capture_output=True).returncode == 0
        if not applied:
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    kit = (ROOT / ".cache/toolchains/cuda-12.8").resolve()
    kit.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "research/phase19-sources/cuda-redist.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(component, [(name, manifest[name]["windows-x86_64"], kit)
                                  for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas", "libcufft")]))
    environment, cmake, ninja, vs = windows_tools()
    environment.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"),
                       CUDA_PATH=str(kit), CUDAToolkit_ROOT=str(kit), VSLANG="1033",
                       PROCESSOR_ARCHITECTURE="AMD64")
    environment["PATH"] = str(kit / "bin") + os.pathsep + environment["PATH"]
    # The installed VS bundle predates newer CUDAToolkit find logic. A local
    # cmake wheel can override it without changing the system installation.
    local_cmake = ROOT / ".cache/toolchains/python/cmake/data/bin/cmake.exe"
    if local_cmake.is_file():
        cmake = str(local_cmake)
    build = ROOT / ".cache/audio-cpp-phase19-sm89-build"
    configure = [
        cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_SYSTEM_NAME=Windows", "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
        f"-DCUDAToolkit_ROOT={kit}", f"-DCMAKE_CUDA_COMPILER={kit / 'bin/nvcc.exe'}",
        "-DCMAKE_CUDA_ARCHITECTURES=89", "-DENGINE_ENABLE_CUDA=ON",
        "-DENGINE_ENABLE_VULKAN=OFF", "-DENGINE_ENABLE_HIP=OFF",
        "-DENGINE_ENABLE_METAL=OFF", "-DENGINE_ENABLE_LLAMAFILE=OFF",
        "-DENGINE_ENABLE_NATIVE_CPU=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON",
        "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=OFF",
        "-DAUDIOCPP_MODEL_SET=custom", "-DAUDIOCPP_MODELS=fish_drama3",
        "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF", "-DENGINE_BUILD_TESTS=OFF",
    ]
    print(json.dumps({"phase": "audio-cpp-configure", "command": configure}), flush=True)
    subprocess.run(configure, env=environment, check=True)
    build_log = ROOT / "research/phase19-audio-cpp-build.log"
    with build_log.open("w", encoding="utf-8") as log:
        built = subprocess.run([cmake, "--build", str(build), "--target", "audiocpp_cli",
                                "--parallel", "4"], env=environment, stdout=log,
                               stderr=subprocess.STDOUT)
    if built.returncode:
        lines = build_log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(line for line in lines if
                        any(marker in line for marker in ("error ", "fatal ", "FAILED:", "Error ")))[:12000],
              flush=True)
        raise RuntimeError(f"audio.cpp build failed; full log: {build_log}")
    executable = build / "bin/audiocpp_cli.exe"
    assert executable.is_file()
    receipt = {
        "source": str(source),
        "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "executable": str(executable), "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cuda_toolkit": str(kit), "vs": vs, "cmake": cmake, "architecture": "SM89",
        "scope": "Isolated benchmark backend; no replacement of the production engine.",
        "local_build_compatibility": "MSVC2019 constexpr lambda capture in unused RMVPE pitch extractor.",
    }
    (ROOT / "research/phase19-audio-cpp-build.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
