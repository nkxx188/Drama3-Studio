"""Build a disk-backed cache of Drama 3's original FP8 routed experts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor" / "fish-speech"
for path in (SRC, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drama_low_vram.expert_cache_file import (  # noqa: E402
    build_expert_cache,
    inspect_expert_cache_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "drama-3",
        help="Official Drama 3 checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache" / "drama3-fp8-experts",
        help="Cache directory on the selected disk",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cache after a complete replacement is built",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0.5,
        help="Free-space reserve required in addition to the cache (default: 0.5)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate source headers and print the planned cache size",
    )
    args = parser.parse_args()
    source = inspect_expert_cache_source(args.checkpoint)
    if args.inspect_only:
        print(json.dumps(source.as_dict(), indent=2))
        return 0

    last_percent = -1

    def progress(copied: int, total: int) -> None:
        nonlocal last_percent
        percent = int(copied * 100 / total) if total else 100
        if percent != last_percent:
            print(f"\rBuilding expert cache: {percent:3d}%", end="", flush=True)
            last_percent = percent

    result = build_expert_cache(
        args.checkpoint,
        args.output,
        force=args.force,
        disk_reserve_bytes=max(0, int(args.reserve_gib * 1024**3)),
        progress=progress,
    )
    print(f"\nExpert cache ready: {result}")
    print(f"Payload: {source.nbytes / 1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
