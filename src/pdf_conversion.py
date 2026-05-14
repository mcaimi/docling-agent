"""Convert PDF files to text records using docling."""

import os
from typing import Any

from docling.document_converter import DocumentConverter

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def convert_pdf_paths_to_documents(pdf_paths: list[str]) -> list[dict[str, Any]]:
    """
    Convert each PDF path to a record ``{"content": str, "metadata": dict}``.

    Failures for individual files are logged to stdout; successful paths are included.
    """
    converter = _get_converter()
    documents: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        try:
            doc = converter.convert(pdf_path)
            text_content = doc.text

            metadata = {
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "page_count": len(doc.pages) if hasattr(doc, "pages") else 0,
                "title": doc.title if hasattr(doc, "title") else "",
            }

            documents.append(
                {
                    "content": text_content,
                    "metadata": metadata,
                }
            )

            print(f"Converted: {os.path.basename(pdf_path)}")

        except Exception as e:
            print(f"Error converting {pdf_path}: {str(e)}")

    return documents
