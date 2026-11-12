# Docling agent pipeline image: one container = one pipeline run (or one shard).
# Configuration comes exclusively from environment variables (see
# src/setup/settings_spec.py); no .env file is baked into the image.
#
#   podman build -t docling-agent -f Containerfile .
#   podman run --rm -v ./pdfs:/data/in:ro,Z -e VECTOR_STORE_BACKEND=none docling-agent
#
# With SOURCE_BACKEND=s3, documents are downloaded one at a time per worker into
# TMPDIR (default /tmp): budget ~MAX_CONCURRENCY x the largest document there.

ARG PYTHON_VERSION=3.11

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# dependencies first, so source changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra vector-stores --extra s3

# docling's default model set, fetched at build time so runs need no
# Hugging Face access and start without a download
RUN /app/.venv/bin/docling-tools models download -o /opt/docling-models

COPY src ./src


FROM python:${PYTHON_VERSION}-slim-bookworm

# shared libraries required by docling's OpenCV-based layout/OCR stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# group 0 + group-writable: runs under an arbitrary UID (OpenShift restricted SCC)
COPY --from=builder --chown=1001:0 /app /app
COPY --from=builder --chown=1001:0 /opt/docling-models /opt/docling-models
RUN chmod -R g=u /app /opt/docling-models

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models \
    FOLDER_PATH=/data/in \
    LOG_FORMAT=json

USER 1001
ENTRYPOINT ["python", "-m", "src.docling_agent"]
