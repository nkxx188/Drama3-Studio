"""Numbered release volumes preserve the complete ZIP and never mix old output."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "split_release_archive", Path(__file__).resolve().parents[1] / "tools/split_release_archive.py")
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


def test_split_release_reassembles_exactly_and_records_each_hash(tmp_path):
    source = tmp_path / "Windows Package.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Windows Package/start.bat", b"launcher" * 100)
    output = tmp_path / "release-parts"
    report = splitter.split_archive(source, output, part_bytes=128)
    parts = [(output / row["name"]).read_bytes() for row in report["parts"]]
    assert len(parts) > 1 and b"".join(parts) == source.read_bytes()
    assert report["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["bytes"] == source.stat().st_size
    for record, content in zip(report["parts"], parts):
        assert 0 < record["bytes"] <= 128
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
    assert (output / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        splitter.split_archive(source, output, part_bytes=128)
    assert b"".join((output / row["name"]).read_bytes() for row in report["parts"]) == source.read_bytes()


def test_invalid_zip_never_creates_a_release_folder(tmp_path):
    source = tmp_path / "unfinished.zip"
    source.write_bytes(b"not a complete archive")
    output = tmp_path / "release-parts"
    with pytest.raises(ValueError, match="completed ZIP"):
        splitter.split_archive(source, output)
    assert not output.exists()
