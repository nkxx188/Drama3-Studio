#!/usr/bin/env bash
# Container entrypoint: mirrors what start.bat does on Windows, then hands over to
# the official launcher. The browser step is skipped because there is no desktop.
set -euo pipefail

ROOT=/app
HOST="${DRAMA_HOST:-0.0.0.0}"
PORT="${DRAMA_PORT:-8879}"

# The worker links against libcublas.so.12 and friends. There is no system CUDA
# toolkit in the image, so point the loader at the copies inside the torch wheels.
SITE="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for subdir in cublas/lib cuda_runtime/lib cufft/lib cuda_nvrtc/lib nvjitlink/lib; do
    candidate="$SITE/nvidia/$subdir"
    if [ -d "$candidate" ]; then
        LD_LIBRARY_PATH="$candidate:${LD_LIBRARY_PATH:-}"
    fi
done
export LD_LIBRARY_PATH

for name in tmp huggingface triton torchinductor torch_extensions cuda; do
    mkdir -p "$ROOT/.cache/$name"
done
mkdir -p "$ROOT/ui_data" "$ROOT/diagnostics"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[drama3] warning: nvidia-smi is missing, the GPU is probably not attached." >&2
    echo "[drama3]          start the container with the compose GPU reservation or --gpus all." >&2
else
    echo "[drama3] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
fi

if [ ! -f "$ROOT/runtime/native/drama3_worker" ]; then
    echo "[drama3] warning: runtime/native/drama3_worker is missing." >&2
    echo "[drama3]          run tools/build_linux_runtime.py and rebuild the image." >&2
fi

if [ -z "$(find "$ROOT/checkpoints" -maxdepth 2 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
    echo "[drama3] note: no .gguf model found under checkpoints." >&2
    echo "[drama3]       mount a model directory, or point Settings at an existing folder." >&2
fi

echo "[drama3] listening on http://${HOST}:${PORT}/  (reach it from the host as <server-ip>:${PORT})"
echo "[drama3] the first model load takes a while; that is expected."

exec /opt/venv/bin/python "$ROOT/tools/windows_launcher.py" \
    --host "$HOST" \
    --port "$PORT" \
    --no-browser
