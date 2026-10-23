"""Build the optional CPU DLL without changing the Python inference stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def windows_tools():
    locator = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    if not locator.is_file():
        raise RuntimeError("Visual Studio Build Tools locator is unavailable")
    installation = subprocess.check_output([
        str(locator), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath",
    ], text=True).strip()
    if not installation:
        raise RuntimeError("Install the x64 MSVC C++ Build Tools to build this optional module")
    installation = Path(installation)
    setup = installation / "VC/Auxiliary/Build/vcvars64.bat"
    result = subprocess.run(
        f'cmd.exe /d /s /c ""{setup}" >nul && set"',
        check=True, capture_output=True, text=True,
    )
    environment = dict(os.environ)
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.startswith("="):
            environment[key] = value
    bundled = installation / "Common7/IDE/CommonExtensions/Microsoft/CMake"
    cmake = shutil.which("cmake", path=environment.get("PATH")) or \
        str(bundled / "CMake/bin/cmake.exe")
    ninja = shutil.which("ninja", path=environment.get("PATH")) or \
        str(bundled / "Ninja/ninja.exe")
    if not Path(cmake).is_file() or not Path(ninja).is_file():
        raise RuntimeError("CMake and Ninja are needed for the native developer build")
    return environment, cmake, ninja, str(installation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/native-build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/native")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Build jobs must be positive")
    build_dir, output_dir = args.build_dir.resolve(), args.output_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    temporary = ROOT / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment.update(TEMP=str(temporary), TMP=str(temporary))
    if os.name == "nt":
        environment, cmake, ninja, installation = windows_tools()
        environment.update(TEMP=str(temporary), TMP=str(temporary))
        extra = [f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    else:
        cmake = shutil.which("cmake")
        ninja = shutil.which("ninja")
        installation = "system C++ compiler"
        if not cmake or not ninja:
            raise RuntimeError("CMake and Ninja are required")
        extra = [f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    subprocess.run([
        cmake, "-S", str(ROOT / "native"), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", *extra,
    ], env=environment, check=True)
    subprocess.run(
        [cmake, "--build", str(build_dir), "--parallel", str(args.jobs)],
        env=environment, check=True,
    )
    filename = "drama_native.dll" if os.name == "nt" else "libdrama_native.so"
    source = build_dir / filename
    if not source.is_file():
        raise RuntimeError(f"Expected native artifact missing: {source}")
    destination = output_dir / filename
    shutil.copy2(source, destination)
    report = {
        "module": str(destination), "build_dir": str(build_dir),
        "toolchain": installation, "cmake": cmake,
        "abi": 1, "scope": "Optional independent CPU C ABI; no PyTorch link or package upgrades.",
        "module_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "sources": {
            path.relative_to(ROOT / "native").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "native").rglob("*"))
            if path.is_file()
        },
    }
    (output_dir / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
