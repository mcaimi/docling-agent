#!/usr/bin/env python
"""Filesystem helpers for discovering PDF files."""

from pathlib import Path
import os

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# list the PDF files in the folder
# return absolute-style string paths for all ``*.pdf`` files in ``folder_path``
def list_pdf_paths(folder_path: str | Path) -> list[str]:
    """Return absolute-style string paths for all ``*.pdf`` files in ``folder_path``."""
    folder = Path(folder_path)
    paths = [str(p) for p in folder.glob("*.pdf")]

    if enable_tracing:
        with _ml.start_span("list_pdfs"):
            _ml.set_tag("folder", str(folder_path))
            _ml.set_tag("pdf_count", len(paths))

    return paths

# export the function
__all__ = ["list_pdf_paths"]