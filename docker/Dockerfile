# Drama 3 Studio on Linux with an NVIDIA GPU.
#
# The build context is the repository root, so both the prepared upstream sources
# and the compiled worker are available:
#
#   python3 tools/prepare_sources.py
#   python3 tools/build_linux_docker_runtime.py    # -> runtime/native-docker/drama3_worker
#   docker compose -f docker/compose.yaml up -d --build
#
# The worker is compiled on the host because it needs a CUDA toolkit; the image only
# needs the shared CUDA libraries, which the torch wheels already ship. Keep
# RUNTIME_IMAGE at least as new as the build host's glibc, otherwise the worker will
# fail to load: a worker built on Ubuntu 26.04 (glibc 2.43) needs a 26.04 runtime,
# while a worker built on Ubuntu 24.04 (glibc 2.39) runs on both.

ARG PYTHON_IMAGE=python:3.12-slim
ARG RUNTIME_IMAGE=ubuntu:26.04

# Ubuntu 26.04 ships Python 3.14, which torch 2.8 does not support. Take the
# interpreter from the official slim image instead; glibc is backward compatible,
# so a Debian-built interpreter runs on the newer runtime.
FROM ${PYTHON_IMAGE} AS python

FROM ${RUNTIME_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin

# Shared libraries the Debian interpreter expects, plus application runtime needs:
# ffmpeg converts reference audio, libsndfile backs soundfile, libgomp1 serves the
# ggml CPU path.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsqlite3-0 libssl3t64 libbz2-1.0 liblzma5 zlib1g libffi8 \
        libncursesw6 libreadline8t64 libuuid1 libexpat1 libtinfo6 \
        ffmpeg libsndfile1 libgomp1 curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python /usr/local /usr/local

RUN /usr/local/bin/python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -U pip wheel setuptools

# torch 2.8.0 on PyPI is the cu128 build; it pulls the nvidia-*-cu12 wheels that
# provide libcublas, libcufft and libcudart for the worker.
RUN /opt/venv/bin/pip install --no-cache-dir --timeout 60 --retries 5 \
        torch==2.8.0 torchaudio==2.8.0

COPY docker/requirements-linux.txt /tmp/requirements-linux.txt
RUN /opt/venv/bin/pip install --no-cache-dir --timeout 60 --retries 5 \
        -r /tmp/requirements-linux.txt \
    && rm -f /tmp/requirements-linux.txt

WORKDIR /app
COPY src/ ./src/
COPY ui/ ./ui/
COPY vendor/fish-speech/ ./vendor/fish-speech/
COPY drama3-mcp/ ./drama3-mcp/
COPY tools/windows_launcher.py ./tools/
COPY docker/entrypoint.sh ./docker/
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY licenses/ ./licenses/
COPY runtime/native-docker/ ./runtime/native/

RUN chmod +x /app/docker/entrypoint.sh /app/runtime/native/drama3_worker \
    && mkdir -p /app/checkpoints /app/ui_data /app/diagnostics /app/.cache

# ui_data is the application state root: jobs, uploads, voices, transcripts and
# settings.json live there, so it has to be a volume in any real deployment.
VOLUME ["/app/checkpoints", "/app/ui_data"]

ENV DRAMA_HOST=0.0.0.0 \
    DRAMA_PORT=8879

EXPOSE 8879

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${DRAMA_PORT}/api/instance" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/docker/entrypoint.sh"]
