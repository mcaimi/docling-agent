#!/usr/bin/env python
"""Filesystem helpers for discovering processable documents."""

from pathlib import Path
import os

# Extensions matched by docling standard pipeline (see document_pipeline.SUPPORTED_INPUT_FORMATS).
_DOCUMENT_GLOBS = ("*.pdf", "*.docx", "*.xlsx", "*.md", "*.html", "*.latex", "*.tex")

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# list supported document files in the folder
def list_document_paths(folder_path: str | Path) -> list[str]:
    """Return paths for PDF, DOCX, XLSX, Markdown, Text, HTML, and LaTeX files in ``folder_path``.

    Kept under the historical name ``list_pdf_paths`` for import compatibility.
    """
    folder = Path(folder_path)
    paths: list[str] = []
    for pattern in _DOCUMENT_GLOBS:
        paths.extend(str(p) for p in folder.glob(pattern))
    paths.sort()

    if enable_tracing:
        with _ml.start_span("list_documents"):
            _ml.set_tag("folder", str(folder_path))
            _ml.set_tag("document_count", len(paths))

    return paths

# export the function
__all__ = ["list_document_paths"]