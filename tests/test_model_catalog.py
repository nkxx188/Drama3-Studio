"""Pinned download choices and safe model-install destinations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from drama_low_vram.model_catalog import download_catalog, get_model
from drama_low_vram.model_download import Downloader
from drama_low_vram.model_installation import ModelInstallation


def test_download_catalog_covers_supported_precision_families_and_channels():
    rows = download_catalog()
    models = {row["id"]: row for row in rows}
    assert {"fp8", "int4", "int8", "w4a8", "w6a8"} <= models.keys()
    assert sum(model["id"].startswith("gguf-") for model in rows) >= 5
    assert set(models["fp8"]["sources"]) == {"huggingface", "modelscope"}
    assert set(models["int4"]["sources"]) == {"huggingface", "modelscope"}
    assert models["int8"]["sources"] == ["huggingface"]
    assert models["w4a8"]["sources"] == ["huggingface"]
    assert models["w6a8"]["sources"] == ["huggingface"]
    assert all(models[row]["sources"] == ["huggingface"]
               for row in models if row.startswith("gguf-"))
    assert all(model["total_bytes"] > 0 for model in rows)


def test_catalog_file_urls_honor_each_pinned_repository(tmp_path):
    int8 = get_model("int8")
    downloader = Downloader(
        tmp_path, int8["files"], 1, 1024,
        source="huggingface", model=int8["repo"], revision=int8["revision"],
        hf_revision=int8["hf_revision"])
    weight = next(row for row in int8["files"] if row["name"].endswith(".safetensors"))
    codec = next(row for row in int8["files"] if row["name"] == "codec.pth")
    assert "PedroMarinhoDev/Drama-3-ComfyUI/resolve/" in downloader.direct_url(weight, "huggingface")
    assert "fishaudio/drama-3/resolve/" in downloader.direct_url(codec, "huggingface")
    assert downloader.direct_url(codec, "huggingface").endswith("/codec.pth?download=true")


def test_bundled_control_head_matches_pinned_catalog_hash():
    model = get_model("int8")
    row = next(row for row in model["files"] if row.get("bundled"))
    path = Path(__file__).parents[1] / "src/drama_low_vram/data" / row["name"]
    assert path.stat().st_size == row["size"]
    with path.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"]


def test_catalog_download_uses_only_its_pinned_checkpoint_directory(tmp_path, monkeypatch):
    import drama_low_vram.model_installation as installation_module

    payload = b"catalog-bound-model" * 1000
    model = {
        "id": "local-test", "label": "Local test", "format": "test", "target": "drama-3-test",
        "kind": "official", "repo": "test/model", "revision": "pinned-revision",
        "hf_revision": "pinned-revision", "sources": ["huggingface"],
        "files": [{
            "name": "weights.bin", "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "huggingface_compatible": True,
        }],
    }
    monkeypatch.setattr(installation_module, "get_model", lambda model_id: model if model_id == "local-test" else None)

    class LocalDownloader(Downloader):
        def run(self):
            self.destination.mkdir(parents=True, exist_ok=True)
            for row in self.files:
                (self.destination / row["name"]).write_bytes(payload)
                self.done_bytes += row["size"]
            self.emit("complete")

    monkeypatch.setattr(installation_module, "Downloader", LocalDownloader)
    checkpoint_root = tmp_path / "checkpoints"
    installer = ModelInstallation(tmp_path / "ui_data", checkpoint_root=checkpoint_root)
    try:
        task = installer.start("download", model_id="local-test", source="huggingface")
        deadline = time.monotonic() + 5
        result = installer.snapshot()
        while result["status"] in ("queued", "running", "cancelling") and time.monotonic() < deadline:
            time.sleep(0.01)
            result = installer.snapshot()
        target = checkpoint_root / "drama-3-test"
        assert task["directory"] == str(target.resolve())
        assert result["status"] == "complete"
        assert (target / "weights.bin").read_bytes() == payload
        receipt = json.loads((target / "download-verification.json").read_text())
        assert receipt["model"] == "test/model"
        assert receipt["files"][0]["verified"] is True
        assert not (tmp_path / "attacker-selected").exists()
    finally:
        installer.close()


def test_comfy_quant_download_builds_a_complete_shared_model_directory(tmp_path, monkeypatch):
    import drama_low_vram.model_installation as installation_module

    source = deepcopy(get_model("int8"))
    source.update(id="local-shared-test", target="drama3-shared-test")
    payloads = {
        "drama3_int8experts_bf16.safetensors": b"packed experts",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "codec.pth": b"codec",
    }
    files = []
    for name, data in payloads.items():
        files.append({
            "name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "huggingface_compatible": True,
        })
    bundled = next(row for row in source["files"] if row.get("bundled"))
    source["files"] = [*files, bundled]
    monkeypatch.setattr(installation_module, "get_model",
                        lambda model_id: source if model_id == source["id"] else None)

    class LocalDownloader(Downloader):
        def run(self):
            self.destination.mkdir(parents=True, exist_ok=True)
            for row in self.files:
                (self.destination / row["name"]).write_bytes(payloads[row["name"]])
                self.done_bytes += row["size"]
            self.emit("complete")

    monkeypatch.setattr(installation_module, "Downloader", LocalDownloader)
    checkpoint_root = tmp_path / "checkpoints"
    installer = ModelInstallation(tmp_path / "ui_data", checkpoint_root=checkpoint_root)
    try:
        installer.start("download", model_id=source["id"], source="huggingface")
        deadline = time.monotonic() + 5
        result = installer.snapshot()
        while result["status"] in ("queued", "running", "cancelling") and time.monotonic() < deadline:
            time.sleep(0.01)
            result = installer.snapshot()
        target = checkpoint_root / source["target"]
        descriptor = json.loads((target / "shared-model.json").read_text())
        assert result["status"] == "complete"
        assert descriptor["model"] == "drama3_int8experts_bf16.safetensors"
        assert descriptor["quantization"] == "int8_convrot"
        assert (target / "control-head.safetensors").is_file()
        assert result["inspection"]["ready"]
    finally:
        installer.close()
