#!/usr/bin/env python
"""Filesystem helpers for discovering PDF files."""

from pathlib import Path

# list the PDF files in the folder
# return absolute-style string paths for all ``*.pdf`` files in ``folder_path``
def list_pdf_paths(folder_path: str | Path) -> list[str]:
    """Return absolute-style string paths for all ``*.pdf`` files in ``folder_path``."""
    folder = Path(folder_path)
    return [str(p) for p in folder.glob("*.pdf")]

# export the function
__all__ = ["list_pdf_paths"]