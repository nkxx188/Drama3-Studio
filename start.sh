#!/usr/bin/env bash
# Drama 3 Studio — portable Linux entry point.
# Keep this window open while Studio is running. Press Ctrl+C to stop.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export DRAMA_NO_PAUSE="${DRAMA_NO_PAUSE:-1}"

PYTHON="$ROOT/.venv-linux/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Drama 3 Studio: the Linux Python environment is missing." >&2
    echo "Install Python 3.12, then run: bash tools/setup-linux.sh" >&2
    exit 1
fi
export DRAMA_PYTHON="$PYTHON"

# Fail early with an actionable hint instead of a long import traceback.
if ! "$PYTHON" -c 'import torch, psutil, soundfile' >/dev/null 2>&1; then
    echo "Drama 3 Studio: Python dependencies are missing or broken." >&2
    echo "Run: bash tools/setup-linux.sh" >&2
    echo "See docs/linux.md for the full Linux setup guide." >&2
    exit 1
fi

exec "$PYTHON" -I "$ROOT/tools/windows_launcher.py" "$@"
