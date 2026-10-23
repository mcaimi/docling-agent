#!/usr/bin/env python
"""Unit tests for result aggregation -- pins the Return Shape documented in AGENTS.md."""

from src.docling_agent import _aggregate_pipeline_results


def test_aggregate_includes_documented_keys():
    agent_settings = {"vector_store_backend": "chroma", "documents": [], "embeddings": []}
    pipeline_result = [
        {
            "path": "/docs/a.pdf",
            "chunks": [{"content": "a", "metadata": {}}],
            "embeddings": [[0.1, 0.2]],
            "error": None,
            "stored_count": 1,
            "upserted_count": 1,
            "retry_count": 2,
            "failed_chunks": [],
        },
        {
            "path": "/docs/b.pdf",
            "chunks": [{"content": "b", "metadata": {}}],
            "embeddings": [None],
            "error": None,
            "store_error": "connection refused",
            "stored_count": 0,
            "failed_chunks": [{"id": "/docs/b.pdf#chunk_0", "error": "connection refused"}],
        },
        {
            "path": "/docs/c.pdf",
            "chunks": [],
            "embeddings": [],
            "error": "conversion failed",
        },
    ]

    result = _aggregate_pipeline_results(agent_settings, pipeline_result)

    # documented Return Shape keys must all be present
    for key in ("documents", "embeddings", "vectors_stored", "vectors_upserted",
                "retry_count", "failed_chunks", "error_reports"):
        assert key in result, f"missing documented key: {key}"

    assert len(result["documents"]) == 2
    assert result["vectors_stored"] == 1
    assert result["vectors_upserted"] == 1
    assert result["retry_count"] == 2
    assert result["failed_chunks"] == [
        {"id": "/docs/b.pdf#chunk_0", "error": "connection refused"}
    ]

    stages = {e["stage"] for e in result["error_reports"]}
    assert stages == {"pipeline", "vector_store"}

    # documents/embeddings from agent_settings are excluded from the spread,
    # everything else passes through
    assert result["vector_store_backend"] == "chroma"


def test_aggregate_handles_missing_optional_keys():
    """Items from skip/error paths don't carry upserted_count/retry_count/failed_chunks."""
    agent_settings = {}
    pipeline_result = [{"path": "/docs/a.pdf", "chunks": [], "embeddings": [], "error": None}]

    result = _aggregate_pipeline_results(agent_settings, pipeline_result)

    assert result["vectors_stored"] == 0
    assert result["vectors_upserted"] == 0
    assert result["retry_count"] == 0
    assert result["failed_chunks"] == []
    assert result["error_reports"] == []
