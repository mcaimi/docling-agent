"""Fixed-size text chunking with overlap for document records."""

from typing import Any


def chunk_document_records(
    documents: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """
    Split each document's ``content`` into overlapping chunks.

    Each output record preserves and extends ``metadata`` with ``chunk_index`` and
    ``chunk_count``.
    """
    chunked_docs: list[dict[str, Any]] = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        chunks: list[str] = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                {
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                }
            )

    return chunked_docs
