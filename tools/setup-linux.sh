#!/usr/bin/env bash
# Prepare a Linux virtual environment and the optional portable CPU core.
# The CUDA native worker is built separately by tools/build_linux_runtime.py.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() { echo "setup-linux: $*" >&2; exit 1; }

find_python() {
    local candidate version
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            case "$version" in
                3.10|3.11|3.12) printf '%s\n' "$candidate"; return 0 ;;
            esac
        fi
    done
    return 1
}

PYTHON="$(find_python || true)"
[[ -n "$PYTHON" ]] || fail "Python 3.10-3.12 is required (Python 3.12 recommended). On Debian/Ubuntu: sudo apt install python3.12 python3.12-venv"

echo "==> Creating virtual environment with $PYTHON"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$PYTHON" -m venv "$ROOT/.venv" || fail "Could not create .venv (install the python3-venv package)"
fi
VENV_PY="$ROOT/.venv/bin/python"

echo "==> Installing pinned Python dependencies (torch cu130 wheels)"
REQUIREMENTS="$ROOT/requirements-linux.lock.txt"
[[ -f "$REQUIREMENTS" ]] || REQUIREMENTS="$ROOT/requirements-linux.txt"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r "$REQUIREMENTS"

if [[ ! -d "$ROOT/vendor/fish-speech/fish_speech" ]]; then
    echo "==> Restoring pinned Fish Speech sources (requires network access)"
    "$VENV_PY" "$ROOT/tools/prepare_sources.py"
else
    echo "==> Pinned Fish Speech sources already present"
fi

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
       .venv/bin/python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
  3. Start Studio:
       ./start.sh

See docs/linux.md for distribution-specific notes and troubleshooting.
EOF
