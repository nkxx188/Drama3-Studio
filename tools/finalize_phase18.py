"""Audit requested deliverables and verify a source-only external ZIP snapshot."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drama_low_vram.model_installation import inspect_checkpoint


def read(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    directories = ["drama-3", "drama-3-int4", "drama-3-int8", "drama-3-w6a8"]
    installed = {name: inspect_checkpoint(ROOT / "checkpoints" / name) for name in directories}
    assert all(row["ready"] for row in installed.values())
    shared = read("shared-model-final-verification.json")
    assert shared["safe_to_remove_redundant_weights"] and shared["all_expert_bytes_equal"]
    results = read("phase18-quantized-results.json")
    for label in ("int4-tuned", "int8-auto", "w6-auto"):
        rows = results["studies"][label]["requests"]
        assert len([r for r in rows if r["case"] == "clone-long" and r["status"] == "complete"]) == 2
        assert any(r["case"] == "clone-long" and r["asr_normalized_text_match"] is True for r in rows)
    assert all(row["codes_equal"] and row["wav_equal"] for row in results["int4_tuning_parity"])
    assert len(results["unconditioned_failures"]) == 2
    for name in ("ui-quantized-models-validation.json", "phase18-ui-browser-audit.json"):
        assert read(name)["passed"], name
    assert read("ui-shared-model-generation.json")["job"]["status"] == "complete"
    suite = ET.parse(ROOT / "research/phase18-pytest.xml")
    assert all(int(row.get("failures", 0)) == 0 and int(row.get("errors", 0)) == 0
               for row in suite.iter("testsuite"))
    retired = read("retired-experiments-manifest.json")
    for row in retired["files"]:
        assert not (ROOT / row["path"]).exists()
        assert sha(Path(retired["archive"]) / row["path"]) == row["sha256"]
    assert not list((ROOT / "checkpoints/drama-3").glob("model-*-of-*.safetensors"))
    assert not (ROOT / ".cache/drama3-fp8-experts/experts.bin").exists()
    assert not (ROOT / ".cache/uv").exists()

    now = datetime.now().astimezone()
    directory = ROOT.parent / "drama-3-backups"
    directory.mkdir(exist_ok=True)
    target = directory / f"drama-3-phase18-code-{now:%Y%m%d-%H%M%S}.zip"
    assert not target.exists()
    reports = [
        "shared-model-final-verification.json", "phase18-quantized-results.json",
        "phase18-storage-audit.json", "retired-experiments-manifest.json",
        "phase18-pytest.xml", "ui-quantized-models-validation.json",
        "phase18-ui-browser-audit.json", "ui-shared-model-generation.json",
        "2026-10-16-shared-models-and-quantized-engine.md",
        "2026-10-16-official-webui-review.md",
    ]
    status = {
        "date": now.isoformat(), "phase": "Shared models and quantized runtime",
        "requested_phase_complete": True, "overall_project_complete": False,
        "training_performed": False, "all_top8_preserved": True,
        "default": "shared official FP8/BF16",
        "model_installations": installed,
        "complete_long_clone_medians": {
            label: study["long_clone_medians"] for label, study in results["studies"].items()
            if study["long_clone_medians"] is not None
        },
        "quality_limits": [
            "INT4 unconditioned short case emitted 33.8s but ASR only recognized hello.",
            "INT8 unconditioned long case reached 768 frames without EOS.",
            "Content ASR does not establish perceptual quality or speaker similarity.",
            "INT4 cold RAM peak is about 37.1GiB; smaller-RAM hardware is unvalidated.",
        ],
        "tests": {"passed": 796, "subtests": 19, "failures": 0},
        "retired_files": len(retired["files"]), "archive": retired["archive"],
        "source_backup": {"path": str(target), "receipt": "research/phase18-backup.json"},
        "evidence_sha256": {name: sha(ROOT / "research" / name) for name in reports},
        "next_project_work": "Broader voice/seed/hardware quality, INT4 load peak and distribution.",
    }
    encoded_status = (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    suffixes = {
        ".py", ".cpp", ".c", ".h", ".hpp", ".md", ".json", ".toml", ".txt",
        ".bat", ".ps1", ".ini", ".yaml", ".yml", ".js", ".cjs", ".css", ".html",
        ".xml", ".png", ".svg", ".jinja", ".ts", ".tsx",
    }
    files = []
    for folder in ("src", "native", "tests", "tools", "ui", "vendor", "research", "checkpoints"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or any(part in ("__pycache__", ".git") for part in path.parts):
                continue
            if path.name in ("native-phase18-status.json", "phase18-backup.json"):
                continue
            if path.suffix not in suffixes and path.name != "control-head.safetensors":
                continue
            files.append(path)
    files.extend(path for path in ROOT.iterdir()
                 if path.is_file() and (path.suffix in suffixes or path.name == ".gitignore"))
    manifest = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            manifest[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            archive.write(path, name)
        name = "research/native-phase18-status.json"
        archive.writestr(name, encoded_status)
        manifest[name] = {"sha256": hashlib.sha256(encoded_status).hexdigest(),
                          "bytes": len(encoded_status)}
        archive.writestr("BACKUP-MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("RESTORE.md",
                        "Restore files into the project directory. Models, .venv, runtime "
                        "caches and audio are excluded. Keep the shared external model files; "
                        "adjust shared-model.json paths if their location changes. "
                        "The final ZIP hash receipt is stored alongside this archive.\n")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, row in manifest.items():
            with archive.open(name) as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"], name
    receipt = {
        "path": str(target), "sha256": sha(target), "files": len(manifest),
        "bytes": target.stat().st_size, "crc_and_all_hashes_verified": True,
        "scope": "Source, tests, UI, vendor, checkpoint metadata/control row and research evidence. "
                 "Excludes model tensors, environments, caches, native binaries and audio.",
    }
    (ROOT / "research/native-phase18-status.json").write_bytes(encoded_status)
    encoded_receipt = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "research/phase18-backup.json").write_text(encoded_receipt, encoding="utf-8")
    target.with_suffix(".receipt.json").write_text(encoded_receipt, encoding="utf-8")
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(encoded_receipt, flush=True)


if __name__ == "__main__":
    main()
