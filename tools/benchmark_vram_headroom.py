"""Run an isolated cache-capacity experiment with a process-local VRAM margin."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import sys

# Import the regular launcher first so all compilation/temp caches stay on the
# project drive. Its normal engine CLI handles the remaining arguments.
from run_drama_engine import main as run_engine


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass normal run_drama_engine.py options after --headroom-mib.",
    )
    parser.add_argument("--headroom-mib", type=int, required=True)
    args, engine_args = parser.parse_known_args()
    if args.headroom_mib < 256:
        parser.error("This experiment retains at least 256 MiB headroom")
    tuning = import_module("drama_low_vram.autotune")
    original = tuning.POST_LOAD_RESERVE_BYTES
    tuning.POST_LOAD_RESERVE_BYTES = args.headroom_mib * 1024**2
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *engine_args]
    print(json.dumps({
        "experiment": "process-local GPU headroom",
        "headroom_mib": args.headroom_mib,
        "execution_workspace_mib": tuning.POST_LOAD_WORKSPACE_BYTES // 1024**2,
        "default_headroom_mib": original // 1024**2,
    }), flush=True)
    try:
        return run_engine()
    finally:
        tuning.POST_LOAD_RESERVE_BYTES = original
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
