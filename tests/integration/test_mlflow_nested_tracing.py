#!/usr/bin/env python
"""Integration tests for per-document nested MLflow tracing.

Uses a real local MLflow tracking backend (sqlite, no external service
needed) because the bug these guard against only manifests with actual
MLflow + mlflow.langchain.autolog() + real LangGraph task concurrency --
mocking MLflow would hide exactly the failure mode being tested for:

- mlflow.start_span_no_context(name) with no explicit parent_span does NOT
  ignore MLflow's own ambient "current span" tracking (verified: it silently
  attaches to whatever span is active, e.g. one created by autolog's own
  instrumentation of the calling LangGraph task).
- When multiple documents' tasks run concurrently and share that ambient
  span, their supposedly-independent per-document root spans used to
  collapse into a single trace instead of one trace per document.

Must be run on its own (e.g. ``pytest tests/integration/test_mlflow_nested_tracing.py``),
not combined with tests/unit in one pytest invocation: document_pipeline.py's
``enable_tracing`` flag is read from ``ENABLE_TRACING`` once at import time, and
``src/graph/__init__.py`` eagerly imports it -- so if any other collected test
file imports anything under ``src.graph`` first, tracing stays disabled here
no matter what this file sets the env var to. Same constraint as every other
tracing-dependent module (see AGENTS.md's MLflow Tracing section).
"""

import os

import mlflow
import pytest

os.environ["ENABLE_TRACING"] = "1"

from src.graph.document_pipeline import document_workflow  # noqa: E402
import src.graph.document_pipeline as document_pipeline_mod  # noqa: E402


def _fake_embed_texts(texts, model, base_url, api_key, dimensions, run_id=None, parent_span=None):
    from src.utils.mlflow_tracing import start_child_span, end_span

    span = start_child_span(run_id, parent_span, "embed_texts")
    end_span(span, model=model, text_count=len(texts))
    return [[0.1, 0.2, 0.3] for _ in texts], []


@pytest.fixture
def sample_markdown(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(3):
        (folder / f"doc{i}.md").write_text(
            f"# Document {i}\n\nSome content for document {i}, long enough "
            f"to produce more than one chunk with a small chunk_size.\n\n"
            f"## Section\n\nMore padding text for document {i}."
        )
    return folder


@pytest.fixture
def mlflow_sqlite_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(document_pipeline_mod, "embed_texts", _fake_embed_texts)
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_name = f"test-nested-tracing-{id(tmp_path)}"
    mlflow.set_experiment(experiment_name)
    return experiment_name


def _run_document_workflow(folder, max_concurrency, run_id, vector_store_backend="none", **config_overrides):
    config = {
        "conversion_pipeline": "standard",
        "chunking_strategy": "fixed",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "text-embedding-ada-002",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key": "sk-fake",
        "embedding_dimensions": None,
        "vector_store_backend": vector_store_backend,
        "mlflow_run_id": run_id,
        "mlflow_session_id": run_id,
        **config_overrides,
    }
    paths = sorted(p for pattern in ("*.md", "*.pdf") for p in folder.glob(pattern))
    input_data = [{"file_path": str(p), "config": config} for p in paths]
    return document_workflow.invoke(
        input_data, config={"configurable": {"max_concurrency": max_concurrency}}
    )


def _pipeline_traces(experiment_name):
    # Trace export happens on a background async queue; without flushing,
    # querying immediately after the workflow returns can race the export
    # and see a trace with some of its later spans still missing.
    mlflow.flush_trace_async_logging()
    exp = mlflow.get_experiment_by_name(experiment_name)
    traces = mlflow.search_traces(locations=[exp.experiment_id])
    return [
        row
        for _, row in traces.iterrows()
        if any(s["name"] == "document_pipeline" for s in row["spans"])
    ]


class TestPerDocumentSpanTree:
    def test_single_document_happy_path_tree(self, sample_markdown, mlflow_sqlite_backend):
        with mlflow.start_run() as run:
            _run_document_workflow(sample_markdown, max_concurrency=1, run_id=run.info.run_id)

        client = mlflow.MlflowClient()
        exp = mlflow.get_experiment_by_name(mlflow_sqlite_backend)
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 1, "no orphan runs should be created"
        assert runs[0].info.status == "FINISHED"

        pipeline_traces = _pipeline_traces(mlflow_sqlite_backend)
        assert len(pipeline_traces) == 3  # one per document

        row = pipeline_traces[0]
        by_id = {s["span_id"]: s for s in row["spans"]}
        root = next(s for s in row["spans"] if s["name"] == "document_pipeline")
        assert root["parent_span_id"] is None, "document_pipeline must be a true root span"
        assert root["status"]["code"] == "STATUS_CODE_OK"

        names_by_parent_name = {
            s["name"]: by_id.get(s["parent_span_id"], {}).get("name")
            for s in row["spans"]
        }
        assert names_by_parent_name["convert_pdf"] == "document_pipeline"
        assert names_by_parent_name["chunk_doc"] == "document_pipeline"
        assert names_by_parent_name["chunk_fixed"] == "chunk_doc"
        assert names_by_parent_name["embed_doc"] == "document_pipeline"
        assert names_by_parent_name["embed_texts"] == "embed_doc"

    def test_concurrent_documents_get_isolated_traces(self, sample_markdown, mlflow_sqlite_backend):
        """Regression test: concurrent documents' root spans used to collapse
        into a single shared trace because start_span_no_context() silently
        attached to an ambient span from autolog's own instrumentation.
        """
        with mlflow.start_run() as run:
            _run_document_workflow(sample_markdown, max_concurrency=3, run_id=run.info.run_id)

        client = mlflow.MlflowClient()
        exp = mlflow.get_experiment_by_name(mlflow_sqlite_backend)
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 1, "no orphan runs should be created"

        pipeline_traces = _pipeline_traces(mlflow_sqlite_backend)
        assert len(pipeline_traces) == 3, (
            "each document must get its own independent trace, not be merged "
            "into a shared one"
        )

        files_seen = set()
        for row in pipeline_traces:
            assert len(row["spans"]) == 6, "each document's trace must have its own full span set"
            root = next(s for s in row["spans"] if s["name"] == "document_pipeline")
            assert root["parent_span_id"] is None
            files_seen.add(root["attributes"]["file"])
        assert len(files_seen) == 3, "each trace must belong to a distinct document"


class TestErrorPathSpanClosure:
    def test_conversion_failure_closes_span_with_error(self, tmp_path, mlflow_sqlite_backend):
        folder = tmp_path / "bad_docs"
        folder.mkdir()
        (folder / "broken.pdf").write_text("not a real pdf")

        with mlflow.start_run() as run:
            _run_document_workflow(folder, max_concurrency=1, run_id=run.info.run_id)

        client = mlflow.MlflowClient()
        exp = mlflow.get_experiment_by_name(mlflow_sqlite_backend)
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 1, "a conversion failure must not leak an orphan run"

        pipeline_traces = _pipeline_traces(mlflow_sqlite_backend)
        assert len(pipeline_traces) == 1
        spans = pipeline_traces[0]["spans"]
        assert len(spans) == 2  # document_pipeline + convert_pdf only
        root = next(s for s in spans if s["name"] == "document_pipeline")
        assert root["status"]["code"] == "STATUS_CODE_ERROR"
        convert = next(s for s in spans if s["name"] == "convert_pdf")
        assert convert["status"]["code"] == "STATUS_CODE_ERROR"

    def test_chunking_failure_closes_span_with_error(self, sample_markdown, mlflow_sqlite_backend):
        with mlflow.start_run() as run:
            _run_document_workflow(
                sample_markdown,
                max_concurrency=1,
                run_id=run.info.run_id,
                chunk_size=10,
                chunk_overlap=50,  # invalid: overlap >= size
            )

        pipeline_traces = _pipeline_traces(mlflow_sqlite_backend)
        row = pipeline_traces[0]
        spans = row["spans"]
        root = next(s for s in spans if s["name"] == "document_pipeline")
        assert root["status"]["code"] == "STATUS_CODE_ERROR"
        convert = next(s for s in spans if s["name"] == "convert_pdf")
        assert convert["status"]["code"] == "STATUS_CODE_OK"
        chunk = next(s for s in spans if s["name"] == "chunk_doc")
        assert chunk["status"]["code"] == "STATUS_CODE_ERROR"
        # no embed_doc/store_doc spans -- the document's journey ended at chunk_doc
        assert not any(s["name"] == "embed_doc" for s in spans)
