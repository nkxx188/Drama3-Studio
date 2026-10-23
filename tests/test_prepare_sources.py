"""Pinned source recovery preserves developer changes and detects drift."""

import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_sources", ROOT / "tools/prepare_sources.py")
sources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


def run(directory, *arguments):
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def fixture_repository(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    run(repository, "init", "--initial-branch=main")
    run(repository, "config", "user.name", "Source fixture")
    run(repository, "config", "user.email", "fixture@example.invalid")
    run(repository, "config", "core.autocrlf", "false")
    (repository / "module.py").write_text("TOP_K = 30\n", encoding="utf-8")
    run(repository, "add", "module.py")
    run(repository, "commit", "-m", "Fixture source")
    commit = run(repository, "rev-parse", "HEAD")
    patch = tmp_path / "sampling.patch"
    patch.write_text(
        "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n"
        "@@ -1 +1 @@\n-TOP_K = 30\n+TOP_K = 0\n",
        encoding="utf-8",
    )
    return repository, commit, patch


def test_clone_pinned_revision_apply_once_and_preserve_unrelated_changes(tmp_path, fixture_repository):
    repository, commit, patch = fixture_repository
    destination = tmp_path / "vendor with spaces/fish-speech"
    assert sources.ensure_checkout(destination, str(repository), commit, patch) == "prepared"
    assert (destination / "module.py").read_text() == "TOP_K = 0\n"
    (destination / "developer-notes.txt").write_text("keep", encoding="utf-8")
    assert sources.ensure_checkout(destination, str(repository), commit, patch) == "already prepared"
    assert (destination / "developer-notes.txt").read_text() == "keep"


def test_conflicting_source_is_rejected_without_reset(tmp_path, fixture_repository):
    repository, commit, patch = fixture_repository
    destination = tmp_path / "vendor"
    sources.ensure_checkout(destination, str(repository), commit, patch)
    (destination / "module.py").write_text("TOP_K = 7\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflict"):
        sources.ensure_checkout(destination, str(repository), commit, patch)
    assert (destination / "module.py").read_text() == "TOP_K = 7\n"


def test_checkout_at_another_revision_is_not_reset(tmp_path, fixture_repository):
    repository, commit, patch = fixture_repository
    destination = tmp_path / "vendor"
    sources.ensure_checkout(destination, str(repository), commit, patch)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        sources.ensure_checkout(destination, str(repository), "0" * 40, patch)
    assert run(destination, "rev-parse", "HEAD") == commit


def test_bundled_patch_matches_its_pinned_checksum():
    metadata, patch = sources.verified_patch(ROOT / "third_party/fish-speech.json")
    assert len(metadata["commit"]) == 40
    assert patch.name == "fish-speech.patch"
    native = json.loads((ROOT / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    native_patch = (ROOT / "native/audio-cpp-runtime.patch").read_bytes()
    assert hashlib.sha256(native_patch).hexdigest() == native["sha256"]
    assert len(native_patch) == native["bytes"]
