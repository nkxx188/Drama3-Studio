"""Read-only logical/unique-file storage audit, accounting for NTFS hardlinks."""

from collections import defaultdict
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def audit(root):
    groups = defaultdict(lambda: {"logical_bytes": 0, "unique_bytes": 0, "files": 0})
    seen, largest = set(), []
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            group = path.relative_to(root).parts[0]
            row = groups[group]
            row["logical_bytes"] += info.st_size
            row["files"] += 1
            identity = info.st_dev, info.st_ino
            if identity not in seen:
                row["unique_bytes"] += info.st_size
                seen.add(identity)
            if info.st_size > 256 * 1024**2:
                largest.append({"path": str(path), "bytes": info.st_size})
    return {
        "root": str(root), "scope": "Logical byte sizes, deduplicated by filesystem file "
        "identity across this tree. Sparse/compressed cluster allocation is not measured.",
        "groups": dict(sorted(groups.items(), key=lambda item: -item[1]["unique_bytes"])),
        "unique_bytes": sum(row["unique_bytes"] for row in groups.values()),
        "logical_bytes": sum(row["logical_bytes"] for row in groups.values()),
        "largest_files": sorted(largest, key=lambda row: -row["bytes"])[:25],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase18-storage-audit.json")
    args = parser.parse_args()
    result = audit(ROOT)
    result["checked_at"] = datetime.now().astimezone().isoformat()
    result["drive_free_bytes"] = shutil.disk_usage(ROOT).free
    result["shared_models"] = [
        json.loads((directory / "shared-model.json").read_text(encoding="utf-8"))
        | {"descriptor_directory": str(directory)}
        for directory in sorted((ROOT / "checkpoints").iterdir())
        if (directory / "shared-model.json").is_file()
    ]
    result["deleted_duplicates"] = {
        "original_shards_bytes": 39_603_574_344,
        "expert_cache_bytes": 32_220_119_040,
        "shards_absent": not list((ROOT / "checkpoints/drama-3").glob("model-*-of-*.safetensors")),
        "expert_cache_absent": not (ROOT / ".cache/drama3-fp8-experts/experts.bin").exists(),
        "uv_cache_absent": not (ROOT / ".cache/uv").exists(),
        "installer_cache_approx_gib": 8.047,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "unique_bytes", "logical_bytes", "drive_free_bytes", "deleted_duplicates")}))


if __name__ == "__main__":
    main()
