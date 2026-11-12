#!/usr/bin/env python
"""Container smoke test: run the pipeline image on a folder of sample docs.

Skipped unless podman is installed and ``DOCLING_AGENT_IMAGE`` names a built
image, e.g.::

    podman build -t docling-agent -f Containerfile .
    DOCLING_AGENT_IMAGE=docling-agent pytest tests/integration/test_container_smoke.py
"""

import json
import os
import shutil
import subprocess

import pytest

IMAGE = os.getenv("DOCLING_AGENT_IMAGE")

pytestmark = pytest.mark.skipif(
    not (IMAGE and shutil.which("podman")),
    reason="needs podman and DOCLING_AGENT_IMAGE",
)


def test_container_processes_folder(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.md").write_text(f"# Document {i}\n\nSample content {i}.\n")
    tmp_path.chmod(0o755)

    proc = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{tmp_path}:/data/in:ro,Z",
            "-e", "VECTOR_STORE_BACKEND=none",
            "-e", "SHARD_INDEX=0",
            "-e", "SHARD_COUNT=2",
            # unreachable embeddings endpoint: per-file failures, still exit 0
            "-e", "EMBEDDING_BASE_URL=http://127.0.0.1:9/v1",
            IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    records = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    messages = [r["msg"] for r in records]
    assert " -> Shard 0/2: 2 of 3 documents" in messages
    assert "Document processing completed successfully" in messages
    assert all(r["shard_index"] == 0 for r in records)


def test_bad_shard_exits_with_config_error():
    proc = subprocess.run(
        ["podman", "run", "--rm", "-e", "SHARD_INDEX=3", "-e", "SHARD_COUNT=2", IMAGE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
