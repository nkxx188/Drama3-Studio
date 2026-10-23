"""Validate Phase19 evidence and create a verified source/evidence snapshot."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drama_low_vram.model_installation import inspect_checkpoint


def read(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    audit = read("phase19-results-audit.json")
    assert all(row["wav_equal"] for row in audit["native_compact_wav_parity"])
    assert len(audit["native_compact_wav_parity"]) == 4
    assert audit["native"]["q4-quality-compact"]["requests"][1]["asr_text_match"]
    assert audit["native"]["q4-quality-compact-plain"]["requests"][0]["asr_text_match"]
    assert read("phase19-ui-quantized-models-validation.json")["passed"]
    assert inspect_checkpoint(ROOT / "checkpoints/drama-3-w4a8")["ready"]
    assert read("phase19-w4a8-download-verified.json")["model"] == "PedroMarinhoDev/Drama-3-ComfyUI"
    assert read("phase19-gguf-q4-quality-download-verified.json")["model"] == "Jmica/Drama-3-GGUF"
    assert read("phase19-gguf-launcher.json") if (ROOT / "research/phase19-gguf-launcher.json").exists() else (
        json.loads((ROOT / "outputs/phase19-gguf-launcher.json").read_text())["complete"])
    suites = ET.parse(ROOT / "research/phase19-pytest.xml")
    assert all(int(s.get("failures", 0)) == 0 and int(s.get("errors", 0)) == 0
               for s in suites.iter("testsuite"))
    with urlopen("http://127.0.0.1:8877/api/model", timeout=10) as response:
        model = json.load(response)
    assert Path(model["selected_directory"]) == ROOT / "checkpoints/drama-3"
    assert model["inspection"]["ready"]
    retired = read("phase19-retired-rotation-manifest.json")
    for row in retired["files"]:
        assert sha(Path(retired["archive"]) / row["path"]) == row["sha256"]
    assert not (ROOT / "tools/probe_rotation_fusion.py").exists()
    assert "rotate_quantize_fused" not in (ROOT / "src/drama_low_vram/quantized_kernels.py").read_text()
    now = datetime.now().astimezone()
    backup_dir = ROOT.parent / "drama-3-backups"
    backup_dir.mkdir(exist_ok=True)
    target = backup_dir / f"drama-3-phase19-code-{now:%Y%m%d-%H%M%S}.zip"
    assert not target.exists()
    evidence = [
        "phase19-results-audit.json", "phase19-pytest.xml",
        "phase19-ui-quantized-models-validation.json", "phase19-storage-audit.json",
        "phase19-audio-cpp-build.json", "phase19-audio-cpp-local.patch",
        "phase19-w4a8-download-verified.json", "phase19-gguf-q4-quality-download-verified.json",
        "phase19-final-asr.json", "2026-10-16-q4-native-and-comfy-tuning.md",
    ]
    status = {
        "date": now.isoformat(), "phase": "Q4 native runtime and Comfy quantized execution",
        "requested_phase_complete": True, "overall_project_complete": False,
        "training_performed": False, "all_top8_preserved": True,
        "default_engine": "Existing PyTorch engine with shared official FP8",
        "added_ui_format": "Shared Comfy W4A8-Codebook/BF16",
        "experimental_native_entry": "run-gguf.bat",
        "native_backend": read("phase19-audio-cpp-build.json"),
        "gguf_selection": read("phase19-gguf-q4-quality-download-pinned.json"),
        "native_long_seconds_before": audit["native"]["q4-quality"]["median_long_request_seconds"],
        "native_long_seconds_after": audit["native"]["q4-quality-compact"]["median_long_request_seconds"],
        "same_gguf_wav_parity": True,
        "native_time_reduction_fraction": audit["native_time_reduction_fraction"],
        "quality_limits": [
            "ASR validates words, not perceptual parity or voice identity against official FP8.",
            "PyTorch W4A8 unconditioned long case capped before finishing.",
            "Native batch missing voice inherits preparation defaults; fresh plain TTS was tested separately.",
            "Comfy output-cached repeats excluded from speed conclusions.",
            "Native CLI cold loading adds latency; no resident UI integration yet.",
            "Other GPUs/RAM configurations unvalidated; native build targets SM89.",
        ],
        "tests": {"passed": 804, "subtests": 19, "failures": 0},
        "ui_restored": True, "ui_port": 8877, "native_default_in_ui": False,
        "failed_rotation_experiment_archive": retired["archive"],
        "source_backup": {"path": str(target), "receipt": "research/phase19-backup.json"},
        "evidence_sha256": {name: sha(ROOT / "research" / name) for name in evidence},
        "next_project_work": "Listening/speaker quality, broader hardware validation and optional "
                             "resident native backend integration with resource admission.",
    }
    status_bytes = (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    suffixes = {".py", ".cpp", ".c", ".h", ".hpp", ".md", ".json", ".toml", ".txt",
                ".bat", ".ps1", ".ini", ".yaml", ".yml", ".js", ".cjs", ".css", ".html",
                ".xml", ".png", ".svg", ".jinja", ".ts", ".tsx", ".patch"}
    files = []
    for folder in ("src", "native", "tests", "tools", "ui", "vendor", "research", "checkpoints"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or any(p in ("__pycache__", ".git") for p in path.parts):
                continue
            if path.name in ("native-phase19-status.json", "phase19-backup.json",
                             "phase19-work-in-progress.json"):
                continue
            if path.suffix in suffixes or path.name == "control-head.safetensors":
                files.append(path)
    files.extend(p for p in ROOT.iterdir() if p.is_file()
                 and (p.suffix in suffixes or p.name == ".gitignore"))
    manifest = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            manifest[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            archive.write(path, name)
        name = "research/native-phase19-status.json"
        archive.writestr(name, status_bytes)
        manifest[name] = {"sha256": hashlib.sha256(status_bytes).hexdigest(), "bytes": len(status_bytes)}
        archive.writestr("BACKUP-MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("RESTORE.md",
                        "Models, .venv, caches, native binaries and audio are excluded. "
                        "Keep existing shared model paths or edit their descriptors. "
                        "To rebuild the native comparison, tools/build_audio_cpp_comparison.py "
                        "uses the pinned upstream source and research/phase19-audio-cpp-local.patch. "
                        "CUDA redistributable metadata is in research/phase19-sources/cuda-redist.json.\n")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, row in manifest.items():
            with archive.open(name) as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"], name
    receipt = {"path": str(target), "sha256": sha(target), "files": len(manifest),
               "bytes": target.stat().st_size, "crc_and_all_hashes_verified": True,
               "scope": "Source, UI, tests, pinned metadata, native patch and research evidence. "
                        "Excludes large weights, environment, toolchain/build caches, native binaries and audio."}
    (ROOT / "research/native-phase19-status.json").write_bytes(status_bytes)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "research/phase19-backup.json").write_text(encoded, encoding="utf-8")
    target.with_suffix(".receipt.json").write_text(encoded, encoding="utf-8")
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    progress = read("phase19-work-in-progress.json")
    progress.update(requested_phase_complete=True, gpu_active=None, remaining=[],
                    completion_status_file="research/native-phase19-status.json")
    (ROOT / "research/phase19-work-in-progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
