"""Print the automatic Drama 3 runtime recommendation for this computer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drama_low_vram.autotune import inspect_and_autotune  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="CUDA index, e.g. cuda:0")
    args = parser.parse_args()
    result = inspect_and_autotune(device=args.device)
    print(result.to_json())
    return 0 if result.config.can_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
