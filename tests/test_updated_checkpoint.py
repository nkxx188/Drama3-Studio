"""The new pinned shard layout is identified independently of user receipts."""
import hashlib
import json
from pathlib import Path

import pytest

from drama_low_vram import model_installation as installation


def test_updated_official_index_selects_new_manifest_and_requires_all_shards(tmp_path):
    document = json.loads(Path(installation.__file__).with_name("data").joinpath(
        "drama3-manifest-20261029.json").read_text(encoding="utf-8"))
    assert len([r for r in document["files"] if r["name"].endswith(".safetensors")]) == 15
    assert all(r["hf_revision"] == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216" for r in document["files"])
    assert installation.official_checkpoint_files(tmp_path) == installation.manifest()


def checkpoint_files(index_payload, shard_name):
    return [
        {"name": "model.safetensors.index.json", "size": len(index_payload),
         "sha256": hashlib.sha256(index_payload).hexdigest()},
        {"name": shard_name, "size": 5, "sha256": hashlib.sha256(b"model").hexdigest()},
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_layout_detection_uses_trusted_hash_not_download_receipt(tmp_path, monkeypatch, legacy):
    current_payload = b'{"weight_map": {"tensor": "new-shard.safetensors"}}'
    old_payload = b'{"weight_map": {"tensor": "old-shard.safetensors"}}'
    current = checkpoint_files(current_payload, "new-shard.safetensors")
    old = checkpoint_files(old_payload, "old-shard.safetensors")
    monkeypatch.setattr(installation, "manifest", lambda *, legacy=False: old if legacy else current)
    payload, expected = (old_payload, old) if legacy else (current_payload, current)
    index = tmp_path / "model.safetensors.index.json"
    index.write_bytes(payload)
    (tmp_path / "download-verification.json").write_text(
        json.dumps({"files": current if legacy else old}), encoding="utf-8")
    assert installation.official_checkpoint_files(tmp_path) == expected
    report = installation.inspect_checkpoint(tmp_path)
    assert report["missing"] == [expected[1]["name"]] and not report["ready"]
    index.write_bytes(payload.replace(b"tensor", b"foobar"))
    assert installation.official_checkpoint_files(tmp_path) == current


def test_unknown_index_size_does_not_trigger_a_full_file_hash(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors.index.json").write_bytes(b"invalid")
    monkeypatch.setattr(installation, "sha256_file", lambda *_args:
                        pytest.fail("Index discovery must reject unexpected sizes before reading the file"))
    assert installation.official_checkpoint_files(tmp_path) == installation.manifest()


@pytest.mark.parametrize("operation", ["verify", "download"])
def test_existing_legacy_installation_keeps_its_snapshot_without_new_shards_or_network(
        tmp_path, monkeypatch, operation):
    import time
    from drama_low_vram.model_download import LEGACY_HF_REVISION, LEGACY_REVISION

    payload = b'{"weight_map": {"tensor": "old-shard.safetensors"}}'
    old = checkpoint_files(payload, "old-shard.safetensors")
    for row in old:
        row.update(revision=LEGACY_REVISION, hf_revision=LEGACY_HF_REVISION)
    current = checkpoint_files(b"new index", "new-shard.safetensors")
    monkeypatch.setattr(installation, "manifest", lambda *, legacy=False: old if legacy else current)
    monkeypatch.setattr(installation.Downloader, "select_source", lambda *_args:
                        pytest.fail("A complete legacy model must not use the network"))
    model = tmp_path / "old-model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_bytes(payload)
    (model / "old-shard.safetensors").write_bytes(b"model")
    installer = installation.ModelInstallation(tmp_path / "state")
    try:
        installer.start(operation, model)
        deadline = time.monotonic() + 5
        while installer.snapshot()["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(.01)
        task = installer.snapshot()
        assert task["status"] == "complete", task
        assert task["inspection"]["ready"]
        assert task["progress"]["total_bytes"] == sum(row["size"] for row in old)
        assert (model / "old-shard.safetensors").read_bytes() == b"model"
        assert not (model / "new-shard.safetensors").exists()
        if operation == "download":
            receipt = json.loads((model / "download-verification.json").read_text(encoding="utf-8"))
            assert receipt["equivalent_huggingface_weight_revision"] == LEGACY_HF_REVISION
    finally:
        installer.close()


def test_older_quantizations_keep_identical_codec_and_tokenizer_components():
    current = {row["name"]: row for row in installation.manifest()}
    old = {row["name"]: row for row in installation.manifest(legacy=True)}
    for name in ("codec.pth", "tokenizer.json"):
        assert current[name]["size"] == old[name]["size"]
        assert current[name]["sha256"] == old[name]["sha256"]
