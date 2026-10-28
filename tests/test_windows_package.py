"""Archive user-data exclusions must not remove dependency modules."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def test_archive_omits_top_level_user_data_but_keeps_dependency_diagnostics(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("testable_windows_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "Portable With Spaces"
    for name in ("ui_data/private.txt", "outputs/private.wav", "diagnostics/private.json",
                 ".cache/private.bin", "checkpoints/nested/drama3.gguf", "logs/private.log",
                 "personal-reference.wav", "extra-models/weights.safetensors", ".env",
                 "ui/.env.local", "runtime/python/Lib/site-packages/example/diagnostics/__init__.py",
                 "runtime/python/Lib/site-packages/example/__pycache__/cached.pyc", "start.bat",
                 "README.md", "README.zh-CN.md"):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    module.write_archive(folder)
    with zipfile.ZipFile(folder.with_suffix(".zip")) as archive:
        names = archive.namelist()
        assert "Portable With Spaces/runtime/python/Lib/site-packages/example/diagnostics/__init__.py" in names
        assert "Portable With Spaces/start.bat" in names
        assert "Portable With Spaces/README.md" in names
        assert "Portable With Spaces/README.zh-CN.md" in names
        assert len(names) == 4


def test_archive_does_not_follow_application_links_to_external_files(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("linked_windows_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "Portable"
    (folder / "ui").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "private.txt").write_text("private", encoding="utf-8")
    try:
        (folder / "ui/linked-file.txt").symlink_to(external / "private.txt")
        (folder / "ui/linked-directory").symlink_to(external, target_is_directory=True)
    except OSError:
        import pytest
        pytest.skip("Creating symbolic links requires Windows Developer Mode or equivalent privileges")
    (folder / "ui/index.html").write_text("application", encoding="utf-8")
    assert list(module.archive_files(folder)) == [folder / "ui/index.html"]


def test_asr_archive_includes_only_explicit_model_files_and_not_speech_weights(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("asr_windows_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "Portable"
    model = folder / module.ASR_DIRECTORY
    model.mkdir(parents=True)
    for name in (*module.ASR_FILES, *module.ASR_NOTICES):
        (model / name).write_bytes(b"fixture")
    for name in ("checkpoints/drama-3-gguf-q4/model.gguf", "checkpoints/other/model.safetensors",
                 "checkpoints/Qwen3-ASR-0.6B-hf/.cache/token", "ui_data/private.wav"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private")
    manifest = folder / "package.json"
    manifest.write_text('{"model_weights_included": false}', encoding="utf-8")
    module.write_archive(folder, include_asr=True)
    with zipfile.ZipFile(folder.with_suffix(".zip")) as archive:
        assert set(archive.namelist()) == {
            "Portable/package.json",
            *(f"Portable/{module.ASR_DIRECTORY.as_posix()}/{name}"
              for name in (*module.ASR_FILES, *module.ASR_NOTICES)),
        }
        bundled = json.loads(archive.read("Portable/package.json"))
        assert bundled["model_weights_included"] is True
        assert bundled["tts_model_weights_included"] is False
        assert bundled["asr_model"]["id"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert json.loads(manifest.read_text()) == {"model_weights_included": False}
    assert (folder / "checkpoints/drama-3-gguf-q4/model.gguf").read_bytes() == b"private"


def test_asr_archive_rejects_incomplete_checkpoint_before_creating_zip(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("incomplete_asr_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "Portable"
    (folder / module.ASR_DIRECTORY).mkdir(parents=True)
    with pytest.raises(ValueError, match="ASR checkpoint is incomplete"):
        module.write_archive(folder, include_asr=True)
    assert not folder.with_suffix(".zip").exists()


def test_archive_omits_torch_link_libraries_and_keeps_runtime_and_python_jit_inputs(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("runtime_only_windows_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "Windows Package"
    kept = (
        "runtime/python/Lib/site-packages/torch/lib/torch_cpu.dll",
        "runtime/python/libs/python312.lib",
        "runtime/python/include/Python.h",
        "runtime/python/Lib/site-packages/triton/backends/nvidia/lib/cuda.lib",
    )
    for name in (*kept, "runtime/python/Lib/site-packages/torch/lib/dnnl.lib",
                 "runtime/python/Lib/site-packages/torch/lib/torch_cuda.lib"):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    assert {file.relative_to(folder).as_posix() for file in module.archive_files(folder)} == set(kept)


def test_refresh_retires_extra_launchers_preserves_data_and_copies_current_app(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "tools/package_windows.py"
    spec = importlib.util.spec_from_file_location("refreshable_windows_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "source"
    package = tmp_path / "Windows Package"
    files = {
        "runtime/native/manifest.json": '{"protocol": 4}',
        "runtime/native/worker.exe": "current worker",
        "src/example.py": "current source",
        "ui/index.html": "current UI",
        "ui/drama3-fish-mark.svg": "<svg/>",
        "vendor/fish-speech/example.py": "vendor source",
        "start.bat": "current launcher",
        "requirements-windows.txt": "dependencies",
        "README.md": "project introduction and installation",
        "README.zh-CN.md": "Chinese introduction and installation",
        "docs/development.md": "source and build instructions",
        "docs/release-preparation.md": "release verification",
        "LICENSE": "MIT for original project material",
        "NOTICE": "upstream notices",
        "THIRD_PARTY_NOTICES.md": "third-party terms",
    }
    files.update({f"tools/{name}": "tool" for name in
                  ("windows_launcher.py", "run_drama_engine.py", "run_drama_gguf.py", "verify_python_runtime.py")})
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for name in (*module.RETIRED_LAUNCHERS, "ui_data/settings.json", "outputs/keep.wav",
                 "checkpoints/keep.gguf", ".cache/keep.bin", *module.RETIRED_UI_ASSETS):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    monkeypatch.setattr(module, "ROOT", project)
    assert module.refresh_application(package)["protocol"] == 4
    assert {path.name for path in package.glob("*.bat")} == {"start.bat"}
    assert (package / "ui/index.html").read_text() == "current UI"
    assert (package / "ui/drama3-fish-mark.svg").read_text() == "<svg/>"
    for name in module.RETIRED_UI_ASSETS:
        assert not (package / name).exists()
    assert (package / "runtime/native/worker.exe").read_text() == "current worker"
    assert "Download a supported speech model from Studio Settings" in (
        package / "START-HERE.txt").read_text(encoding="utf-8")
    for name in ("README.md", "README.zh-CN.md", "docs/development.md", "docs/release-preparation.md",
                 "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert (package / name).read_text(encoding="utf-8") == (project / name).read_text(encoding="utf-8")
    for name in ("ui_data/settings.json", "outputs/keep.wav", "checkpoints/keep.gguf", ".cache/keep.bin"):
        assert (package / name).read_bytes() == b"keep"
