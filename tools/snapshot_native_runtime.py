"""Save the complete generated vendor delta required to reproduce the shipped worker."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / ".cache/audio-cpp-phase19"
    delta = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=source).decode().splitlines()
    for name in untracked:
        path = source / name
        if path.is_file():
            patch = f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
            patch += "".join(difflib.unified_diff(
                [], path.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="/dev/null", tofile="b/" + name))
            delta += patch.encode("utf-8")
    target = ROOT / "native/audio-cpp-runtime.patch"
    target.write_bytes(delta)
    metadata = ROOT / "native/audio-cpp-runtime.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value.update(sha256=hashlib.sha256(delta).hexdigest(), bytes=len(delta), native_protocol=5)
    value["features"] = ["ordinary", "references", "previous_takes", "stream", "infill",
                         "audio_stream", "design", "joint_dialogue", "reference_prewarm"]
    metadata.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "apply", "--reverse", "--check", str(target)], cwd=source, check=True)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
