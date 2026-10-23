#!/usr/bin/env python
"""Fixed-size and Hybrid text chunking with overlap for document records."""

from typing import Any

# import HybridChunker from Docling
try:
    from docling.chunking import HybridChunker
except ImportError as e:
    raise e

import os

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# fixed-size chunker
def chunk_document_records(
    documents: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """
    Split each document's ``content`` into overlapping chunks.

    Each output record preserves and extends ``metadata`` with ``chunk_index`` and
    ``chunk_count``.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive or ``chunk_overlap`` is not smaller than
        ``chunk_size``. Without this check, ``start`` never advances past the end
        of ``content`` and the loop below never terminates.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

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

    if enable_tracing:
        with _ml.start_span("chunk_fixed"):
            _ml.set_tag("chunk_size", chunk_size)
            _ml.set_tag("chunk_overlap", chunk_overlap)
            _ml.set_tag("total_chunks", len(chunked_docs))

    return chunked_docs


# HybridChunker using docling APIs
def chunk_document_records_hybrid(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # chunking results
    chunked_docs: list[dict[str, Any]] = []

    # the chunker object
    chunker = HybridChunker()

    # loop over documents
    for doc in documents:
        content = doc["docling_document"]
        metadata = doc["metadata"]

        # generate chunks using hybrid chunker
        chunks = list(chunker.chunk(dl_doc=content))
        for i, chunk in enumerate(chunks):
            # contextualize chunk for content storage
            chunk_content = chunker.contextualize(chunk=chunk)

            chunked_docs.append(
                {
                    "content": chunk_content,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                }
            )

    if enable_tracing:
        with _ml.start_span("chunk_hybrid"):
            _ml.set_tag("total_chunks", len(chunked_docs))

    return chunked_docs
