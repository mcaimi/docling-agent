#!/usr/bin/env python
"""Filesystem helpers for discovering processable documents."""

import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Extensions matched by docling standard pipeline (see document_pipeline.SUPPORTED_INPUT_FORMATS).
# Published in the settings contract, so launchers count documents the same way.
DOCUMENT_GLOBS = ("*.pdf", "*.docx", "*.xlsx", "*.md", "*.html", "*.latex", "*.tex")

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        logger.warning("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# list supported document files in the folder
def list_document_paths(folder_path: str | Path) -> list[str]:
    """Return paths for PDF, DOCX, XLSX, Markdown, Text, HTML, and LaTeX files in ``folder_path``.

    Kept under the historical name ``list_pdf_paths`` for import compatibility.
    """
    folder = Path(folder_path)
    paths: list[str] = []
    for pattern in DOCUMENT_GLOBS:
        paths.extend(str(p) for p in folder.glob(pattern))
    paths.sort()

    if enable_tracing:
        with _ml.start_span("list_documents"):
            _ml.set_tag("folder", str(folder_path))
            _ml.set_tag("document_count", len(paths))

    return paths


# select this instance's share of the documents
def shard_paths(paths: list[str], shard_index: int, shard_count: int) -> list[str]:
    """Return every ``shard_count``-th path starting at ``shard_index``.

    Deterministic and non-overlapping across ``shard_index in range(shard_count)``
    as long as every shard sees the same ``paths`` in the same order
    (``list_document_paths`` sorts), so parallel instances (e.g. pods of a
    Kubernetes Indexed Job) split the work without a coordinator.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    return paths[shard_index::shard_count]

# export the functions
__all__ = ["DOCUMENT_GLOBS", "list_document_paths", "shard_paths"]
