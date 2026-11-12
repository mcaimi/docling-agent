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

Tracing-dependent modules read ``ENABLE_TRACING`` once at import time (see
AGENTS.md's MLflow Tracing section), and ``src/graph/__init__.py`` eagerly
imports them -- so when another collected test file imports anything under
``src.graph`` first, tracing would stay disabled here no matter what this file
sets the env var to. The autouse ``tracing_enabled`` fixture switches those
module-level flags on per test, so this file also runs combined with tests/unit.
"""

import json
import os

import mlflow
import pytest

os.environ["ENABLE_TRACING"] = "1"

from src.graph.document_pipeline import document_workflow  # noqa: E402
import src.graph.document_pipeline as document_pipeline_mod  # noqa: E402
import src.graph.document_paths as document_paths_mod  # noqa: E402
import src.graph.vector_store as vector_store_mod  # noqa: E402

# Modules whose ``enable_tracing`` flag is frozen at import time.
_TRACED_MODULES = (document_pipeline_mod, document_paths_mod, vector_store_mod)


@pytest.fixture(autouse=True)
def tracing_enabled(monkeypatch):
    """Enable tracing even if the modules were imported before ENABLE_TRACING=1."""
    for mod in _TRACED_MODULES:
        monkeypatch.setattr(mod, "enable_tracing", True)
        monkeypatch.setattr(mod, "_ml", mlflow, raising=False)


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


class TestRunStatusDuringPipeline:
    def test_run_stays_running_until_owner_ends_it(self, sample_markdown, mlflow_sqlite_backend):
        """Regression test: run_context() used to re-enter the run with
        ``with mlflow.start_run(run_id=...)``, whose exit calls end_run() --
        marking the run FINISHED on the server as soon as the first worker
        task returned, while the pipeline was still running.
        """
        client = mlflow.MlflowClient()
        with mlflow.start_run() as run:
            _run_document_workflow(sample_markdown, max_concurrency=3, run_id=run.info.run_id)
            assert client.get_run(run.info.run_id).info.status == "RUNNING"
            # the owning thread still has its run active
            assert mlflow.active_run().info.run_id == run.info.run_id

        assert client.get_run(run.info.run_id).info.status == "FINISHED"


class TestShardChildRun:
    def test_shard_logs_child_run_progress_and_session_traces(self, sample_markdown, mlflow_sqlite_backend):
        """A UI-launched shard: nested child of a scheduler-created parent run,
        live step metrics, run summary, and per-document traces grouped under
        the parent's session."""
        from src.docling_agent import process_documents
        from src.setup.agent_settings import _get_settings_from_environment

        client = mlflow.MlflowClient()
        exp = mlflow.get_experiment_by_name(mlflow_sqlite_backend)
        # the scheduler creates the parent without making it active anywhere
        parent = client.create_run(exp.experiment_id, run_name="logical-run")
        parent_id = parent.info.run_id

        settings = {
            **_get_settings_from_environment(),
            "folder_path": str(sample_markdown),
            "chunk_size": 100,
            "chunk_overlap": 10,
            "embedding_api_key": "sk-fake",
            "max_concurrency": 1,
            "shard_index": 0,
            "shard_count": 2,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "mlflow_experiment_name": mlflow_sqlite_backend,
            "mlflow_parent_run_id": parent_id,
            "mlflow_session_id": None,
            "scheduler_run_id": "sched-1",
            "scheduler_executor": "podman",
        }
        result = process_documents(settings, enable_tracing=True)

        child_id = result["mlflow_run_id"]
        child = client.get_run(child_id)
        assert child.info.status == "FINISHED"
        assert child.data.tags["mlflow.parentRunId"] == parent_id
        assert child.data.tags["shard_index"] == "0"
        assert child.data.tags["shard_count"] == "2"
        assert child.data.tags["scheduler_run_id"] == "sched-1"
        assert child.data.tags["total_document_count"] == "3"
        assert result["mlflow_session_id"] == parent_id  # defaults to the parent
        # the shard must not touch the parent's lifecycle
        assert client.get_run(parent_id).info.status == "RUNNING"

        # shard 0 of 2 over 3 sorted docs -> doc0, doc2; one batch each
        history = client.get_metric_history(child_id, "docs_completed")
        assert [(m.step, m.value) for m in history] == [(1, 1), (2, 2)]
        assert child.data.metrics["document_count"] == 2
        assert child.data.metrics["total_chunks"] == len(result["documents"])

        artifacts = {a.path for a in client.list_artifacts(child_id)}
        assert "run_summary.json" in artifacts
        summary = mlflow.artifacts.load_dict(f"runs:/{child_id}/run_summary.json")
        assert summary["status"] == "completed"
        assert summary["document_count"] == 2

        mlflow.flush_trace_async_logging()
        traces = mlflow.search_traces(
            locations=[exp.experiment_id],
            filter_string=f"metadata.`mlflow.trace.session` = '{parent_id}'",
        )
        roots = [
            next(s for s in row["spans"] if s["name"] == "document_pipeline")
            for _, row in traces.iterrows()
            if any(s["name"] == "document_pipeline" for s in row["spans"])
        ]
        # span attributes come back JSON-encoded in the search_traces frame
        attrs = [{k: json.loads(v) for k, v in r["attributes"].items()} for r in roots]
        assert sorted(a["file"] for a in attrs) == ["doc0.md", "doc2.md"]
        for a in attrs:
            assert a["shard_index"] == 0
            assert a["file_format"] == ".md"
            assert a["chunk_count"] > 0
            assert a["stored_count"] == 0


class TestConvertSpanTiming:
    def test_convert_span_covers_conversion(self, sample_markdown, mlflow_sqlite_backend, monkeypatch):
        """Regression test: the document and convert_pdf spans used to start
        after converter.convert() returned, so conversion time was missing
        from the trace (convert_pdf always ~0 ms)."""
        import time as _time

        real_new_converter = document_pipeline_mod._new_converter

        class SlowConverter:
            def __init__(self, inner):
                self.inner = inner

            def convert(self, path):
                _time.sleep(0.25)
                return self.inner.convert(path)

        monkeypatch.setattr(
            document_pipeline_mod, "_new_converter", lambda **kw: SlowConverter(real_new_converter(**kw))
        )
        with mlflow.start_run() as run:
            _run_document_workflow(sample_markdown, max_concurrency=1, run_id=run.info.run_id)

        for row in _pipeline_traces(mlflow_sqlite_backend):
            convert = next(s for s in row["spans"] if s["name"] == "convert_pdf")
            root = next(s for s in row["spans"] if s["name"] == "document_pipeline")
            assert (int(convert["end_time_unix_nano"]) - int(convert["start_time_unix_nano"])) >= 250_000_000
            assert int(root["start_time_unix_nano"]) <= int(convert["start_time_unix_nano"])
