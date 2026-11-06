"""Pinned download choices and safe model-install destinations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Event
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from drama_low_vram.model_catalog import download_catalog, get_model
from drama_low_vram.model_download import Downloader, HF_REVISION, manifest
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


def test_official_catalog_and_default_downloader_use_the_same_updated_snapshot():
    model = get_model("fp8")
    assert model["hf_revision"] == HF_REVISION == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
    assert model["target"] == "drama-3-20261029-fp8"
    assert model["files"] == manifest()
    assert sum(row["name"].endswith(".safetensors") for row in model["files"]) == 15


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_updated_official_urls_pin_both_channels_to_verified_files(tmp_path, source):
    model = get_model("fp8")
    downloader = Downloader(tmp_path, model["files"], 1, 1024, source=source,
                            model=model["repo"], revision=model["revision"],
                            hf_revision=model["hf_revision"])
    for row in model["files"]:
        url = downloader.direct_url(row, source)
        if source == "huggingface":
            assert url.startswith(f"https://huggingface.co/fishaudio/drama-3/resolve/{HF_REVISION}/")
        else:
            assert url.startswith("https://modelscope.cn/api/v1/models/fishaudio/drama-3/repo?")
            assert parse_qs(urlsplit(url).query) == {
                "Revision": [row["revision"]], "FilePath": [row["name"]],
            }
            assert len(row["revision"]) == 40


@pytest.mark.parametrize("model_id,size,digest", [
    ("gguf-4bit-experts-q4-k-s", 23_371_670_016,
     "5f71c8b52dc50ceaeb4c1fc99ba30db1333c0fbbaa4d6934fa6c7c7a3c9d3f62"),
    ("gguf-8bit-f32-routers-bf16-embed-head-experts-q4-k-s", 25_634_025_024,
     "415e20ee598f51409864517b8e8b5efd51e50c45583a37c8ffaa70952fa15bd5"),
    ("gguf-8bit-f32-routers-experts-q4-k-s", 24_665_239_072,
     "d0da72ffdbba62965c92653ed40d0851ccc80f335de02bc7d8a7c61b305c9011"),
    ("gguf-8bit-experts-q2-k", 17_053_647_360,
     "5cb86588eff6404f4482830cfd3d69713f27115c02530a29f772535262fb7826"),
    ("gguf-8bit-experts-q3-k-s", 20_325_204_480,
     "c3f86b7e831e05769f21561950d6604d48edda684aa123b7ca763314fc5a6223"),
    ("gguf-8bit-experts-q4-k-s", 24_603_394_560,
     "4ffc1028e0133497a551c5661a6b709639c4ac0da38b8d421ee51f52ab4b0574"),
])
def test_gguf_catalog_pins_v2_weights_in_separate_directories(tmp_path, model_id, size, digest):
    model = get_model(model_id)
    assert model["revision"] == model["hf_revision"] == "060acc8e5d9e5d16baf647c775dfe1a1438f5f35"
    assert model["target"] == model_id + "-20261029"
    row, = model["files"]
    assert row["size"] == size and row["sha256"] == digest
    downloader = Downloader(tmp_path, model["files"], 1, 1024, source="huggingface",
                            model=model["repo"], revision=model["revision"],
                            hf_revision=model["hf_revision"])
    assert f"/resolve/{model['hf_revision']}/" in downloader.direct_url(row, "huggingface")


@pytest.mark.parametrize("model_id,old_target", [
    ("fp8", "drama-3-fp8"),
    ("gguf-8bit-experts-q4-k-s", "gguf-8bit-experts-q4-k-s"),
])
def test_updated_catalog_does_not_resume_in_or_overwrite_old_directory(tmp_path, model_id, old_target):
    root = tmp_path / "checkpoints"
    old = root / old_target
    old.mkdir(parents=True)
    (old / "model.bin").write_bytes(b"keep previous weights")
    (old / ".download-state.json").write_bytes(b"keep previous partial download")
    installer = ModelInstallation(tmp_path / "state", checkpoint_root=root)
    release = Event()
    installer._executor.submit(lambda: release.wait(timeout=10))
    try:
        task = installer.start("download", model_id=model_id, source="huggingface")
        assert Path(task["directory"]) == (root / get_model(model_id)["target"]).resolve()
        assert Path(task["directory"]) != old.resolve()
        assert installer.cancel()["status"] == "cancelled"
        assert (old / "model.bin").read_bytes() == b"keep previous weights"
        assert (old / ".download-state.json").read_bytes() == b"keep previous partial download"
        assert not Path(task["directory"]).exists()
    finally:
        release.set()
        installer.close()


def test_unrebuilt_quantizations_keep_their_original_matching_components():
    assert get_model("int4")["hf_revision"] == "a6f588691a13b6607cb09f9800556487ab7aa0fe"
    for model_id in ("int8", "w6a8", "w4a8"):
        model = get_model(model_id)
        assert model["hf_revision"] == "9063c072019aa578895e51adb5489c6cd2c834b0"
        config = next(row for row in model["files"] if row["name"] == "config.json")
        assert config["sha256"] == "4b13a31b504cf54181f52df5bbb27604f5cc4b77b6e90a12ab85d62e0d70b693"
        assert config["hf_revision"] == "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"


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
