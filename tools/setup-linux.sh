#!/usr/bin/env bash
# Prepare a Linux virtual environment and the optional portable CPU core.
# The CUDA native worker is built separately by tools/build_linux_runtime.py.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() { echo "setup-linux: $*" >&2; exit 1; }

python_is_312() {
    "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1
}

VENV="$ROOT/.venv-linux"
VENV_PY="$VENV/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    PYTHON=""
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_is_312 "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON" ]] || fail "Install Python 3.12 with venv support. See docs/linux.md."
    echo "==> Creating .venv-linux with $PYTHON"
    "$PYTHON" -m venv "$VENV" || fail "Could not create .venv-linux (check your Python venv installation)"
fi
python_is_312 "$VENV_PY" || fail ".venv-linux must use Python 3.12. Move the old environment aside and run setup again."

echo "==> Installing pinned Python dependencies (torch cu130 wheels)"
export PIP_CACHE_DIR="$ROOT/.cache/pip-linux"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r "$ROOT/requirements-linux.lock.txt"

echo "==> Checking pinned Fish Speech sources"
"$VENV_PY" "$ROOT/tools/prepare_sources.py"

if command -v cmake >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
    echo "==> Building optional portable CPU core (libdrama_native.so)"
    "$VENV_PY" "$ROOT/tools/build_native.py" || echo "Optional CPU core build failed; GPU inference is unaffected."
else
    echo "==> Skipping optional CPU core (install cmake and ninja to build it)"
fi

cat <<'EOF'

Setup finished. Next step: build the CUDA native worker.

  1. Install a CUDA 13.x toolkit (nvcc), if it is not already present.
  2. Run:
       .venv-linux/bin/python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
  3. Start Studio:
       ./start.sh

See docs/linux.md for distribution-specific notes and troubleshooting.
EOF
