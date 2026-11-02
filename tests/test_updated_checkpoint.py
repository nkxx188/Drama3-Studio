"""The new pinned shard layout is identified independently of user receipts."""
import hashlib
import json
from pathlib import Path

from drama_low_vram import model_installation as installation


def test_updated_official_index_selects_new_manifest_and_requires_all_shards(tmp_path):
    document = json.loads(Path(installation.__file__).with_name("data").joinpath(
        "drama3-manifest-20261029.json").read_text(encoding="utf-8"))
    assert len([r for r in document["files"] if r["name"].endswith(".safetensors")]) == 15
    assert all(r["hf_revision"] == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216" for r in document["files"])
    assert installation.official_checkpoint_files(tmp_path) == installation.manifest()


def test_new_layout_detection_uses_trusted_hash_not_download_receipt(tmp_path, monkeypatch):
    payload = b'{"weight_map": {"tensor": "new-shard.safetensors"}}'
    index = tmp_path / "model.safetensors.index.json"
    index.write_bytes(payload)
    document = {"files": [
        {"name": index.name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        {"name": "new-shard.safetensors", "size": 5, "sha256": "0" * 64},
    ]}
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs:
        json.dumps(document) if path.name == "drama3-manifest-20261029.json" else original(path, **kwargs))
    assert installation.official_checkpoint_files(tmp_path) == document["files"]
    report = installation.inspect_checkpoint(tmp_path)
    assert report["missing"] == ["new-shard.safetensors"] and not report["ready"]
    index.write_bytes(payload.replace(b"tensor", b"foobar"))
    assert installation.official_checkpoint_files(tmp_path) == installation.manifest()
