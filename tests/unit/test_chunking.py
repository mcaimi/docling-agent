#!/usr/bin/env python
"""Unit tests for fixed-size chunking validation."""

import pytest
from src.graph.chunking import chunk_document_records


def _doc(content: str) -> dict:
    return {"content": content, "metadata": {"file_path": "/test/doc.txt"}}


class TestChunkDocumentRecordsValidation:
    """chunk_overlap must be smaller than chunk_size, or `start` never advances."""

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError):
            chunk_document_records([_doc("x" * 50)], chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            chunk_document_records([_doc("x" * 50)], chunk_size=10, chunk_overlap=20)

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_document_records([_doc("x" * 50)], chunk_size=0, chunk_overlap=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_document_records([_doc("x" * 50)], chunk_size=-5, chunk_overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_document_records([_doc("x" * 50)], chunk_size=10, chunk_overlap=-1)


class TestChunkDocumentRecordsValidInputs:
    def test_valid_overlap_produces_chunks(self):
        chunks = chunk_document_records([_doc("x" * 25)], chunk_size=10, chunk_overlap=5)
        assert len(chunks) > 0
        assert all(c["metadata"]["chunk_count"] == len(chunks) for c in chunks)

    def test_zero_overlap_is_allowed(self):
        chunks = chunk_document_records([_doc("x" * 20)], chunk_size=10, chunk_overlap=0)
        assert len(chunks) == 2

    def test_chunk_index_sequential(self):
        chunks = chunk_document_records([_doc("x" * 30)], chunk_size=10, chunk_overlap=2)
        assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
