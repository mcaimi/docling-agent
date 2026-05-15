#!/usr/bin/env python
"""Fixed-size and Hybrid text chunking with overlap for document records."""

from typing import Any

# import HybridChunker from Docling
try:
    from docling.chunking import HybridChunker
except ImportError as e:
    raise e


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
            # fill metadata
            c_metadata = {
                "name": chunk.meta.origin.filename,
                "uri": chunk.meta.origin.uri,
                "headings": chunk.meta.headings,
                "captions": chunk.meta.captions,
                "mimetype": chunk.meta.origin.mimetype,
                "document_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}",
                "chunk_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}_chunk_{i}",
            }
            # fill chunk metadata
            chunk_metadata = {
                "document_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}",
                "chunk_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}_chunk_{i}",
                "source": c_metadata.get("uri") or c_metadata.get("name"),
            }

            chunked_docs.append(
                {
                    "chunk_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}_chunk_{i}",
                    "content": chunk_content,
                    "chunk_metadata": chunk_metadata,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                }
            )

    # return chunked_docs
    return chunked_docs
