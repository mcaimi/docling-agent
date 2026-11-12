#!/usr/bin/env python
"""Unit tests for per-batch progress metrics and MLflow param redaction."""

from unittest.mock import patch

from src.docling_agent import _aggregate_pipeline_results, _is_sensitive_param
from src.graph import document_pipeline
from src.graph.document_pipeline import progress_metrics

RESULTS = [
    {
        "path": "/docs/a.pdf",
        "chunks": [{"content": "a"}, {"content": "b"}],
        "embeddings": [[0.1], None],
        "error": None,
        "embedding_error": "1/2 chunks failed to embed",
        "stored_count": 1,
        "upserted_count": 1,
        "retry_count": 2,
        "failed_chunks": [],
    },
    {
        "path": "/docs/b.pdf",
        "chunks": [{"content": "c"}],
        "embeddings": [[0.2]],
        "error": None,
        "store_error": "connection refused",
        "stored_count": 0,
        "failed_chunks": [{"id": "/docs/b.pdf#chunk_0", "error": "connection refused"}],
    },
    {"path": "/docs/c.pdf", "chunks": [], "embeddings": [], "error": "conversion failed"},
]


def test_progress_metrics_counts():
    assert progress_metrics(RESULTS) == {
        "docs_completed": 3,
        "docs_failed": 2,
        "total_chunks": 3,
        "total_embeddings": 3,
        "non_null_embeddings": 2,
        "vectors_stored": 1,
        "vectors_upserted": 1,
        "vector_store_retry_count": 2,
        "failed_chunk_count": 1,
        "error_count": 3,
        "vector_store_error_count": 1,
    }


def test_progress_metrics_agree_with_aggregation():
    metrics = progress_metrics(RESULTS)
    final = _aggregate_pipeline_results({}, RESULTS)
    assert metrics["total_chunks"] == len(final["documents"])
    assert metrics["total_embeddings"] == len(final["embeddings"])
    assert metrics["vectors_stored"] == final["vectors_stored"]
    assert metrics["vectors_upserted"] == final["vectors_upserted"]
    assert metrics["vector_store_retry_count"] == final["retry_count"]
    assert metrics["failed_chunk_count"] == len(final["failed_chunks"])
    assert metrics["error_count"] == len(final["error_reports"])


def test_log_progress_uses_explicit_run_id_and_step():
    with patch("mlflow.MlflowClient") as client_cls:
        document_pipeline._log_progress("run-123", RESULTS[:2])
    run_id = client_cls.return_value.log_batch.call_args.args[0]
    metrics = client_cls.return_value.log_batch.call_args.kwargs["metrics"]
    assert run_id == "run-123"
    assert {m.step for m in metrics} == {2}
    assert {m.key: m.value for m in metrics}["docs_completed"] == 2


def test_log_progress_is_noop_without_run_id():
    with patch("mlflow.MlflowClient") as client_cls:
        document_pipeline._log_progress(None, RESULTS)
    client_cls.assert_not_called()


def test_log_progress_never_raises():
    with patch("mlflow.MlflowClient", side_effect=RuntimeError("server down")):
        document_pipeline._log_progress("run-123", RESULTS)


def test_sensitive_params_are_redacted():
    for key in ("api_key", "embedding_api_key", "vlm_api_headers", "vlm_api_params",
                "milvus_token", "some_new_password"):
        assert _is_sensitive_param(key), key
    for key in ("chunk_size", "folder_path", "mlflow_parent_run_id", "user_prompt"):
        assert not _is_sensitive_param(key), key
