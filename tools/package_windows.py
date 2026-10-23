"""Assemble a self-contained Windows folder from the working pinned environment.

No model payloads, user data, build toolchains, virtualenv paths or research
outputs are copied. Python is based on the redistributable standalone runtime,
with installed distribution records copied through their dependency closure.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
RETIRED_LAUNCHERS = (
    "run-ui.bat", "run-engine.bat", "run-gguf.bat", "run-doctor.bat",
    "run-drama3.bat", "setup-windows.bat",
)
RETIRED_UI_ASSETS = ("ui/fish-audio-logo.svg", "ui/audio-mark.svg")
PACKAGE_DIRECTORIES = frozenset(("src", "ui", "vendor", "tools", "runtime", "licenses", "docs"))
PACKAGE_FILES = frozenset((
    "start.bat", "requirements-windows.txt", "START-HERE.txt", "package.json",
    "README.md", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSE", "LICENSE.txt",
))


def copy_changed(source, destination):
    source, destination = Path(source), Path(destination)
    if (not destination.is_file() or source.stat().st_size != destination.stat().st_size
            or source.stat().st_mtime_ns != destination.stat().st_mtime_ns):
        shutil.copy2(source, destination)
    return str(destination)


def refresh_application(destination):
    """Update the app in an existing runtime without touching models or user data."""
    native = ROOT / "runtime/native"
    if not (native / "manifest.json").is_file():
        raise RuntimeError("Build the universal NVIDIA runtime before completing the package")
    shutil.copytree(native, destination / "runtime/native", dirs_exist_ok=True, copy_function=copy_changed)
    optional = ROOT / ".cache/native/drama_native.dll"
    if optional.is_file():
        copy_changed(optional, destination / "runtime/native/drama_native.dll")
    for name in ("src", "ui", "vendor/fish-speech"):
        shutil.copytree(ROOT / name, destination / name, dirs_exist_ok=True, copy_function=copy_changed,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "checkpoints",
                                                     "outputs", ".cache", ".venv", "node_modules", "editor-source"))
    for name in RETIRED_LAUNCHERS:
        # Only known generated entry points are retired; user data is never cleaned.
        (destination / name).unlink(missing_ok=True)
    for name in RETIRED_UI_ASSETS:
        # Retire only known project branding assets, never user files.
        (destination / name).unlink(missing_ok=True)
    for name in ("start.bat", "requirements-windows.txt"):
        copy_changed(ROOT / name, destination / name)
    for name in ("README.md", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSE", "LICENSE.txt"):
        if (ROOT / name).is_file():
            copy_changed(ROOT / name, destination / name)
    for name in ("development.md", "release-preparation.md"):
        source = ROOT / "docs" / name
        if source.is_file():
            (destination / "docs").mkdir(exist_ok=True)
            copy_changed(source, destination / "docs" / name)
    if (ROOT / "licenses").is_dir():
        shutil.copytree(ROOT / "licenses", destination / "licenses",
                        dirs_exist_ok=True, copy_function=copy_changed)
    for name in ("windows_launcher.py", "run_drama_engine.py", "run_drama_gguf.py", "verify_python_runtime.py"):
        target = destination / "tools" / name
        target.parent.mkdir(exist_ok=True)
        copy_changed(ROOT / "tools" / name, target)
    (destination / "checkpoints").mkdir(exist_ok=True)
    (destination / "START-HERE.txt").write_text(
        "Drama 3 Windows NVIDIA\n\n"
        "1. Extract the complete folder to a writable local SSD path.\n"
        "2. Double-click start.bat. The Studio opens in your browser.\n"
        "3. Select your model in Studio. Keep the startup window open while using it.\n\n"
        "start.bat is the only startup entry. It selects the bundled Python, checks the "
        "environment, reopens an existing instance, and chooses another local port if needed.\n"
        "If startup fails, the window keeps the error visible. Details are saved under diagnostics/.\n"
        "The window shows service, model and task status only. Detailed output is saved to "
        "diagnostics/studio.log with bounded rotation.\n"
        "Support diagnostics can be exported from Studio Settings. If Studio cannot open, "
        "run start.bat --diagnostics in a terminal.\n\n"
        "Python, CUDA runtime libraries and required Python packages are included. "
        "No separate Python, CUDA Toolkit, WSL or Visual Studio installation is required.\n"
        "A compatible NVIDIA driver and an x64/AVX2 CPU are required. "
        "RTX20/30/40/50 code targets are included; physical validation is on RTX4070 Laptop.\n\n"
        "Model weights are separate. Download a supported model from Studio Settings, or place "
        "model files under this folder's checkpoints directory and select them in Studio. You "
        "can also add another model folder from Settings. Formats and precision remain your choice.\n"
        "Cache, outputs, settings and logs stay in this folder. User audio and settings are preserved "
        "when refreshing application files. Externally stored models need to remain accessible.\n"
        "Reference audio may be up to 120 seconds, subject to context and memory capacity.\n",
        encoding="utf-8")
    return json.loads((native / "manifest.json").read_text(encoding="utf-8"))


def runtime_distributions():
    pending = []
    for line in (ROOT / "requirements-windows.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "--")):
            pending.append(Requirement(line))
    seen, result = set(), {}
    while pending:
        request = pending.pop()
        if request.marker and not request.marker.evaluate():
            continue
        key = canonicalize_name(request.name)
        extras = tuple(sorted(request.extras))
        if (key, extras) in seen:
            continue
        seen.add((key, extras))
        distribution = metadata.distribution(request.name)
        if request.specifier and not request.specifier.contains(distribution.version, prereleases=True):
            raise RuntimeError(f"Installed {request.name} {distribution.version} differs from {request.specifier}")
        result[key] = distribution
        for dependency in distribution.requires or ():
            dependency = Requirement(dependency)
            if not dependency.marker or any(dependency.marker.evaluate({"extra": extra}) for extra in ("", *extras)):
                dependency.marker = None
                pending.append(dependency)
    return result


def copy_distribution(distribution, target):
    source = Path(distribution.locate_file("")).resolve()
    for relative in distribution.files or ():
        original = Path(distribution.locate_file(relative)).resolve()
        if not original.is_relative_to(source) or not original.is_file():
            continue
        relative = original.relative_to(source)
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        if relative.parts[:2] == ("torch", "lib") and relative.suffix.lower() == ".lib":
            continue
        # Editable source-path injections and developer virtualenv hooks must
        # not make the delivered interpreter find the developer's checkout.
        if relative.suffix == ".pth":
            content = original.read_text(encoding="utf-8", errors="replace")
            if "__editable__" in content or any(Path(line).is_absolute() for line in content.splitlines()
                                               if line and not line.startswith(("import ", "#"))):
                continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != original.stat().st_size:
            shutil.copy2(original, destination)


def archive_files(destination):
    """Only release application files, without traversing user model locations."""
    destination = Path(destination).resolve()
    for folder, directories, filenames in os.walk(destination, followlinks=False):
        folder = Path(folder)
        directories[:] = sorted(
            name for name in directories
            if (folder != destination or name in PACKAGE_DIRECTORIES)
            and name not in (".git", "__pycache__", "node_modules", ".pytest_cache")
            and not (folder / name).is_symlink()
            and not (folder / name).is_junction()
        )
        for name in sorted(filenames):
            file = folder / name
            relative = file.relative_to(destination)
            if (folder == destination and name not in PACKAGE_FILES
                    or name.endswith((".pyc", ".pyo"))
                    or name == ".env" or name.startswith(".env.")
                    or (relative.parts[:6] == ("runtime", "python", "Lib", "site-packages", "torch", "lib")
                        and file.suffix.lower() == ".lib")
                    or file.is_symlink()
                    or not file.resolve().is_relative_to(destination)):
                continue
            yield file


def write_archive(destination):
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for file in archive_files(destination):
            archive.write(file, file.relative_to(destination.parent))
    with archive_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    archive_path.with_suffix(".zip.sha256").write_text(digest + "\n", encoding="ascii")
    print("Archive:", archive_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Drama3 Windows NVIDIA")
    parser.add_argument("--prepare-python-only", action="store_true")
    parser.add_argument("--refresh-app", action="store_true",
                        help="Update code, UI, launcher and native runtime in an existing package")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ROOT or not destination.is_relative_to((ROOT / "dist").resolve()):
        raise ValueError("Package output must be a dedicated folder inside this project's dist directory")
    if args.archive_only:
        if not (destination / "package.json").is_file():
            raise RuntimeError("Complete the package folder before creating its archive")
        write_archive(destination)
        return
    if args.refresh_app:
        python = destination / "runtime/python"
        dependency_file = python / "runtime-dependencies.json"
        if not (python / "python.exe").is_file() or not dependency_file.is_file():
            raise RuntimeError("Prepare the complete Python runtime before refreshing application files")
        native = refresh_application(destination)
        manifest_file = destination / "package.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
        manifest.update({
            "schema_version": 1, "platform": "windows-x86_64", "model_weights_included": False,
            "entry_point": "start.bat", "native": native,
            "formats": ["FP8/BF16", "GPTQ-W4A16", "INT8-ConvRot", "W6A8", "W4A8", "Drama3-GGUF"],
            "dependencies": json.loads(dependency_file.read_text(encoding="utf-8")),
        })
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.zip:
            write_archive(destination)
        print("Updated application:", destination, flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    python = destination / "runtime/python"
    python.mkdir(parents=True, exist_ok=True)
    base = Path(sys.base_prefix)
    for original in base.iterdir():
        # Triton's bundled C compiler needs Python.h and python312.lib at
        # runtime; omitting them makes a copied Python import but fail at JIT.
        if original.name in ("Scripts", "__pycache__", "pyvenv.cfg"):
            continue
        if original.is_dir():
            shutil.copytree(original, python / original.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "site-packages"))
        else:
            shutil.copy2(original, python / original.name)
    distributions = runtime_distributions()
    site = python / "Lib/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for name, distribution in sorted(distributions.items()):
        print("Bundling", name, distribution.version, flush=True)
        copy_distribution(distribution, site)
    (python / "runtime-dependencies.json").write_text(
        json.dumps({name: value.version for name, value in sorted(distributions.items())}, indent=2) + "\n",
        encoding="utf-8")
    if args.prepare_python_only:
        return
    native = refresh_application(destination)
    manifest = {
        "schema_version": 1, "platform": "windows-x86_64", "model_weights_included": False,
        "entry_point": "start.bat",
        "formats": ["FP8/BF16", "GPTQ-W4A16", "INT8-ConvRot", "W6A8", "W4A8", "Drama3-GGUF"],
        "native": native,
        "python": sys.version, "dependencies": {name: value.version for name, value in distributions.items()},
    }
    (destination / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.zip:
        write_archive(destination)
    print("Package:", destination, flush=True)


if __name__ == "__main__":
    main()
