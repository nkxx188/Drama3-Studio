"""Restore pinned Fish Speech sources and the local sampling patch, without weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *arguments, check=True):
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and result.returncode:
        raise RuntimeError(f"Git failed in {directory}: {result.stderr.strip()}")
    return result


def _patch_new_files(patch):
    """List files the patch creates; they are reliable 'already applied' markers."""
    names = []
    current = None
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[-1]
        elif line.startswith("new file mode ") and current:
            names.append(current)
    return names


def ensure_checkout(destination, repository, commit, patch):
    """Apply a pinned patch once; refuse to reset an existing checkout."""
    destination, patch = Path(destination).resolve(), Path(patch).resolve()
    if not patch.is_file():
        raise FileNotFoundError(patch)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", repository, str(destination)],
            check=True,
        )
        git(destination, "checkout", "--detach", commit)
    elif not (destination / ".git").exists():
        raise RuntimeError(f"Existing source folder is not a Git checkout: {destination}")
    actual = git(destination, "rev-parse", "HEAD").stdout.strip()
    if actual != commit:
        raise RuntimeError(
            f"Source revision mismatch in {destination}: expected {commit}, got {actual}. "
            "Keep or move your existing checkout before preparing a separate pinned copy."
        )
    if git(destination, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        return "already prepared"
    # A checkout copied from Windows may carry the patch plus later research
    # edits and CRLF line endings. A strict reverse check then fails even though
    # the patch is present. Files created by the patch cannot exist otherwise.
    new_files = _patch_new_files(patch)
    if new_files and all((destination / name).is_file() for name in new_files):
        return "already prepared"
    check = git(destination, "apply", "--check", str(patch), check=False)
    if check.returncode:
        raise RuntimeError(
            f"Local changes conflict with the pinned patch in {destination}. "
            f"No files were reset.\n{check.stderr.strip()}"
        )
    git(destination, "apply", "--whitespace=nowarn", str(patch))
    git(destination, "apply", "--reverse", "--check", str(patch))
    return "prepared"


def verified_patch(metadata_file):
    metadata_file = Path(metadata_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    patch = (metadata_file.parent / metadata["patch"]).resolve()
    if not patch.is_relative_to(metadata_file.parent.resolve()):
        raise ValueError("Patch must be next to its pinned source metadata")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError(f"Source patch checksum mismatch: {patch}")
    return metadata, patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    metadata, patch = verified_patch(ROOT / "third_party/fish-speech.json")
    status = ensure_checkout(
        ROOT / "vendor/fish-speech", metadata["repository"], metadata["commit"], patch,
    )
    print(f"Fish Speech {metadata['commit']}: {status}")


if __name__ == "__main__":
    main()
