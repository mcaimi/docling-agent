#!/usr/bin/env python
"""Per-document parallel processing tasks using langgraph.func API."""

import logging
import os
import time
from typing import Any, Literal

from langgraph.types import RunnableConfig
from langgraph.func import entrypoint, task

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.models.inference_engines.vlm.base import VlmEngineType
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from pydantic import AnyUrl
except ImportError as e:
    raise ImportError(f"docling is required: {e}") from e

try:
    from .chunking import chunk_document_records, chunk_document_records_hybrid
    from .embeddings import embed_texts
    from .vector_store import get_vector_store, store_chunk_embeddings
    from ..utils.mlflow_tracing import (
        session_context,
        run_context,
        start_document_span,
        start_child_span,
        end_span,
    )
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# Formats supported by the standard docling pipeline (VLM remains PDF-only).
SUPPORTED_INPUT_FORMATS: tuple[InputFormat, ...] = (
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.XLSX,
    InputFormat.MD,
    InputFormat.HTML,
    InputFormat.LATEX,
)

logger = logging.getLogger(__name__)

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        logger.warning("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

def _file_path_from_item(item: dict) -> str:
    """Resolve document path from pipeline input (``file_path`` or legacy ``pdf_path``)."""
    path = item.get("file_path") or item.get("pdf_path")
    if not path:
        raise KeyError("item must include 'file_path' or 'pdf_path'")
    return path


def _is_pdf_path(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() == ".pdf"


def _effective_pipeline(
    file_path: str, pipeline: Literal["standard", "vlm"]
) -> Literal["standard", "vlm"]:
    """VLM conversion applies only to PDF; other extensions use the standard pipeline."""
    if pipeline == "vlm" and not _is_pdf_path(file_path):
        return "standard"
    return pipeline


# create a DocumentConverter instance per parallel execution
def _new_converter(
    pipeline: Literal["standard", "vlm"] = "standard",
    vlm_api_url: str | None = None,
    vlm_api_headers: dict[str, str] | None = None,
    vlm_api_params: dict[str, Any] | None = None,
    vlm_preset: str | None = None,
) -> DocumentConverter:
    """Create a fresh DocumentConverter instance (no singleton caching)."""
    if pipeline == "vlm":
        if not vlm_api_url:
            raise ValueError("VLM pipeline requires vlm_api_url=")
        vlm_options = VlmConvertOptions.from_preset(
            vlm_preset or "smoldocling",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API,
                url=AnyUrl(vlm_api_url),
                headers=vlm_api_headers or {},
                params=vlm_api_params or {},
            ),
        )
        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True,
            vlm_options=vlm_options,
        )
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

    return DocumentConverter(allowed_formats=list(SUPPORTED_INPUT_FORMATS))


# convert docling data to dictionary
def _result_to_record(
    file_path: str,
    conv: ConversionResult,
    conversion_pipeline: Literal["standard", "vlm"],
) -> dict[str, Any]:
    """Enrich a conversion result into a record with content and metadata."""
    text_content = conv.document.export_to_markdown()
    page_count = conv.input.page_count or len(conv.pages)
    title = ""
    try:
        title = conv.input.file.stem
    except (AttributeError, TypeError):
        title = os.path.splitext(os.path.basename(file_path))[0]

    metadata: dict[str, Any] = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "page_count": page_count,
        "title": title,
        "conversion_pipeline": conversion_pipeline,
    }
    return {
        "docling_document": conv.document,
        "content": text_content,
        "metadata": metadata,
    }

# LANGGRAPH TASKS

@task
def convert_doc(
    item: dict,
) -> dict:
    """Convert a single document with a dedicated DocumentConverter instance.

    Supports PDF, DOCX, XLSX, and Markdown. VLM pipeline applies to PDF only;
    other formats are converted with the standard pipeline.

    Parameters
    ----------
    item : {"file_path": str, "config": dict} or {"pdf_path": str, "config": dict}
        Contains the file path and agent config.

    Returns
    -------
    dict
        {"doc": record|None, "path": str, "error": str|None, "config": dict}
    """
    config = item["config"]
    file_path = _file_path_from_item(item)
    requested_pipeline = config.get("conversion_pipeline")
    pipeline = _effective_pipeline(file_path, requested_pipeline)
    session_id = config.get("mlflow_session_id") if enable_tracing else None
    # Passed to run_context() below: LangGraph runs this task in a worker thread
    # whose context doesn't inherit the active run from the main thread, so
    # without re-entering it explicitly each span here would spawn its own
    # orphan MLflow run.
    run_id = config.get("mlflow_run_id") if enable_tracing else None

    file_name = os.path.basename(file_path)
    file_format = os.path.splitext(file_path)[1].lower()
    # Root-span attributes: let the scheduler build its per-document view
    # from traces alone.
    doc_attributes = {
        "file": file_name,
        "file_format": file_format,
        "shard_index": config.get("shard_index", 0),
    }

    logger.info(
        " -> conversion task in path %s, using pipeline mode: %s%s",
        file_name,
        pipeline,
        f" (VLM skipped for non-PDF; requested {requested_pipeline})"
        if pipeline != requested_pipeline
        else "",
        extra={"file": file_name, "stage": "convert"},
    )

    with session_context(session_id), run_context(run_id):
        # Root span for this document's whole pipeline journey (convert ->
        # chunk -> embed -> store). It isn't context-managed because its
        # lifetime spans multiple independently-scheduled LangGraph tasks;
        # each later stage closes it on its own last exit path -- see
        # chunk_doc/embed_doc/store_doc below. Both spans start before the
        # conversion so its duration is part of the trace.
        doc_span = start_document_span(run_id, "document_pipeline", **doc_attributes)
        convert_span = start_child_span(run_id, doc_span, "convert_pdf")
        try:
            converter = _new_converter(
                pipeline=pipeline,
                vlm_api_url=config.get("vlm_api_url"),
                vlm_api_headers=config.get("vlm_api_headers"),
                vlm_api_params=config.get("vlm_api_params"),
                vlm_preset=config.get("vlm_preset"),
            )

            result = converter.convert(file_path)
            doc = _result_to_record(file_path, result, pipeline)
            end_span(
                convert_span,
                pipeline=pipeline,
                file=file_name,
                file_format=file_format,
                page_count=result.input.page_count,
            )

            return {
                "doc": doc,
                "path": file_path,
                "error": None,
                "config": config,
                "mlflow_doc_span": doc_span,
            }
        except Exception as e:
            end_span(convert_span, error=str(e))
            end_span(doc_span, error=str(e), stage_failed="convert")
            logger.error(
                "Error converting %s: %s", file_path, e,
                extra={"file": file_name, "stage": "convert"},
            )
            return {
                "doc": None,
                "path": file_path,
                "error": str(e),
                "config": config,
                "mlflow_doc_span": None,
            }


@task
def chunk_doc(
    item: dict,
) -> dict:
    """Chunk one document.

    Parameters
    ----------
    item : dict with "converted" (Task) and "config" keys.

    Returns
    -------
    dict
        {"chunks": [...], "path": str, "error": str|None, "config": dict}
    """
    converted_task = item["converted"]
    result = converted_task.result()
    config = result.get("config", {})
    session_id = config.get("mlflow_session_id") if enable_tracing else None
    run_id = config.get("mlflow_run_id") if enable_tracing else None

    with session_context(session_id), run_context(run_id):
        if result.get("error") or result["doc"] is None:
            # Upstream (convert_doc) already closed the document span (or
            # never created one) -- nothing to do with it here.
            return {
                "chunks": [],
                "path": result["path"],
                "error": result["error"],
                "config": config,
                "mlflow_doc_span": None,
            }

        strategy = config.get("chunking_strategy")
        doc = result["doc"]
        doc_span = result.get("mlflow_doc_span")
        chunk_span = start_child_span(run_id, doc_span, "chunk_doc")

        try:
            if strategy == "hybrid":
                chunks = chunk_document_records_hybrid(
                    [doc], run_id=run_id, parent_span=chunk_span
                )
            else:
                chunks = chunk_document_records(
                    [doc],
                    config.get("chunk_size", 1000),
                    config.get("chunk_overlap", 100),
                    run_id=run_id,
                    parent_span=chunk_span,
                )
        except Exception as e:
            end_span(chunk_span, error=str(e))
            end_span(doc_span, error=str(e), stage_failed="chunk")
            logger.error(
                "Error chunking %s: %s", result["path"], e,
                extra={"file": os.path.basename(result["path"]), "stage": "chunk"},
            )
            return {
                "chunks": [],
                "path": result["path"],
                "error": str(e),
                "config": config,
                "mlflow_doc_span": None,
            }

        end_span(chunk_span, chunking_strategy=strategy, chunk_count=len(chunks))

        logger.info(
            " -> Chunking completed for path %s: produced %d chunks",
            result.get("path"),
            len(chunks),
            extra={"file": os.path.basename(result["path"]), "stage": "chunk"},
        )
        return {
            "chunks": chunks,
            "path": result["path"],
            "error": None,
            "config": config,
            "mlflow_doc_span": doc_span,
        }


@task
def embed_doc(
    item: dict,
) -> dict:
    """Embed one document's chunks.

    Parameters
    ----------
    item : dict with "chunked" (Task) and "config" keys.

    Returns
    -------
    dict
        {"chunks": [...], "embeddings": [...], "path": str, "error": str|None,
        "embedding_error": str|None, "config": dict}

        ``error`` means the whole document has nothing to store (upstream
        failure or no chunks); ``embedding_error`` is a non-blocking note that
        some (but not necessarily all) chunks failed to embed after retries --
        ``chunks``/``embeddings`` still carry whatever succeeded.
    """
    chunked_task = item["chunked"]
    result = chunked_task.result()
    config = result.get("config", {})
    session_id = config.get("mlflow_session_id") if enable_tracing else None
    run_id = config.get("mlflow_run_id") if enable_tracing else None

    with session_context(session_id), run_context(run_id):
        if result.get("error") or not result["chunks"]:
            doc_span = result.get("mlflow_doc_span")
            # If this is a genuine upstream error, chunk_doc already closed
            # (or never created) the document span. If instead chunk_doc
            # succeeded with zero chunks, there's nothing left for store_doc
            # to do either, so close the document's trace here, cleanly.
            if doc_span is not None and not result.get("error"):
                end_span(doc_span, chunk_count=0, stored_count=0)
            return {
                "embeddings": [],
                "path": result["path"],
                "error": result["error"],
                "embedding_error": None,
                "config": config,
                "mlflow_doc_span": None,
            }

        chunks = result["chunks"]
        texts = [c["content"] for c in chunks]
        doc_span = result.get("mlflow_doc_span")
        embed_span = start_child_span(run_id, doc_span, "embed_doc")

        log_extra = {"file": os.path.basename(result["path"]), "stage": "embed"}
        logger.info(" -> Embedding %d chunks...", len(texts), extra=log_extra)

        try:
            vectors, embed_errors = embed_texts(
                texts,
                config.get("embedding_model"),
                config.get("embedding_base_url"),
                config.get("embedding_api_key", ""),
                config.get("embedding_dimensions"),
                run_id=run_id,
                parent_span=embed_span,
            )
            failed_count = sum(1 for v in vectors if v is None)
            embedding_error = (
                f"{failed_count}/{len(vectors)} chunks failed to embed: "
                + "; ".join(embed_errors)
                if failed_count
                else None
            )
            end_span(
                embed_span,
                error=embedding_error,
                embedding_model=config.get("embedding_model", "unknown"),
                chunk_count=len(texts),
                embedding_count=len(vectors),
                non_null_count=sum(1 for v in vectors if v is not None),
            )
        except Exception as e:
            end_span(embed_span, error=str(e))
            end_span(doc_span, error=str(e), stage_failed="embed")
            raise

        logger.info(" -> Produced %d embeddings.", len(vectors), extra=log_extra)
        if embedding_error:
            logger.warning(" -> %s", embedding_error, extra=log_extra)
        return {
            "chunks": chunks,
            "embeddings": vectors,
            "path": result["path"],
            "error": None,
            "embedding_error": embedding_error,
            "config": config,
            "mlflow_doc_span": doc_span,
        }


@task
def store_doc(item: dict) -> dict:
    """Persist one document's chunk embeddings to the configured vector store.

    Parameters
    ----------
    item : dict with "embedded" (Task) key.

    Returns
    -------
    dict
        Embed output plus ``stored_count`` and ``store_error``.
    """
    embedded_task = item["embedded"]
    result = embedded_task.result()
    config = result.get("config", {})
    session_id = config.get("mlflow_session_id") if enable_tracing else None
    run_id = config.get("mlflow_run_id") if enable_tracing else None
    backend = (config.get("vector_store_backend") or "none").strip().lower()
    doc_span = result.get("mlflow_doc_span")
    log_extra = {"file": os.path.basename(result["path"]), "stage": "store"}
    # Attributes set on the document's root span on every exit path below.
    doc_summary = {
        "chunk_count": len(result.get("chunks", [])),
        "embedding_failed": bool(result.get("embedding_error")),
    }

    logger.debug(" -> Vector store backend: %s", backend, extra=log_extra)

    base = {
        "chunks": result.get("chunks", []),
        "embeddings": result.get("embeddings", []),
        "path": result["path"],
        "error": result.get("error"),
        "embedding_error": result.get("embedding_error"),
        "config": config,
        "stored_count": 0,
        "store_error": None,
    }

    with session_context(session_id), run_context(run_id):
        # This is the last stage for this document -- every exit path below
        # must close doc_span exactly once, since nothing runs after it.
        if (
            backend in ("", "none")
            or result.get("error")
            or not result.get("chunks")
        ):
            end_span(
                doc_span, error=result.get("error"), stored_count=0, **doc_summary
            )
            return base

        logger.info(
            " -> Storing embeddings for %s (%s)...",
            os.path.basename(result["path"]),
            backend,
            extra=log_extra,
        )
        store_span = start_child_span(run_id, doc_span, "store_doc")
        try:
            vector_store = get_vector_store(config)
            if vector_store is None:
                end_span(store_span)
                end_span(doc_span, stored_count=0, **doc_summary)
                return base

            batch_size = max(1, int(config.get("vector_store_batch_size", 200)))
            write_mode = config.get("vector_store_write_mode", "insert")
            max_workers = max(1, int(config.get("vector_store_max_batch_workers", 4)))

            store_result = store_chunk_embeddings(
                vector_store,
                result["chunks"],
                result["embeddings"],
                batch_size=batch_size,
                write_mode=write_mode,
                max_workers=max_workers,
                config=config,
                run_id=run_id,
                parent_span=store_span,
            )
            stored_count = store_result["stored_count"]
            skipped = store_result["skipped_count"]
            upserted = store_result.get("upserted_count", 0)
            retry_count = store_result.get("retry_count", 0)
            failed_chunks = store_result.get("failed_chunks", [])

            end_span(
                store_span,
                backend=backend,
                file=os.path.basename(result["path"]),
                write_mode=write_mode,
                stored_count=stored_count,
                skipped_count=skipped,
                upserted_count=upserted,
                retry_count=retry_count,
                failed_count=len(failed_chunks),
            )
            end_span(
                doc_span,
                stored_count=stored_count,
                failed_chunk_count=len(failed_chunks),
                **doc_summary,
            )

            logger.info(
                " -> Stored %d vectors (%d skipped, %d upserted, %d retries, %d failed).",
                stored_count,
                skipped,
                upserted,
                retry_count,
                len(failed_chunks),
                extra=log_extra,
            )
            return {
                **base,
                "stored_count": stored_count,
                "upserted_count": upserted,
                "retry_count": retry_count,
                "failed_chunks": failed_chunks,
            }
        except Exception as e:
            end_span(store_span, backend=backend, error=str(e))
            end_span(
                doc_span, error=str(e), stage_failed="store", stored_count=0, **doc_summary
            )
            logger.error(
                "Error storing vectors for %s: %s", result["path"], e, extra=log_extra
            )
            return {**base, "store_error": str(e)}


def _max_concurrency_from_config(config: RunnableConfig) -> int:
    """Read and validate max_concurrency from RunnableConfig.configurable."""
    raw = (config or {}).get("configurable", {}).get("max_concurrency")
    if raw is None:
        return os.cpu_count() or 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return os.cpu_count() or 1


def progress_metrics(results: list[dict]) -> dict[str, int]:
    """Cumulative counters over the per-document results produced so far.

    Names match the end-of-run metrics, so the latest logged value of each is
    also the final one.
    """
    metrics = {
        "docs_completed": len(results),
        "docs_failed": 0,
        "total_chunks": 0,
        "total_embeddings": 0,
        "non_null_embeddings": 0,
        "vectors_stored": 0,
        "vectors_upserted": 0,
        "vector_store_retry_count": 0,
        "failed_chunk_count": 0,
        "error_count": 0,
        "vector_store_error_count": 0,
    }
    for r in results:
        embeddings = r.get("embeddings", [])
        metrics["docs_failed"] += bool(r.get("error") or r.get("store_error"))
        metrics["total_chunks"] += len(r.get("chunks", []))
        metrics["total_embeddings"] += len(embeddings)
        metrics["non_null_embeddings"] += sum(1 for e in embeddings if e is not None)
        metrics["vectors_stored"] += r.get("stored_count", 0)
        metrics["vectors_upserted"] += r.get("upserted_count", 0)
        metrics["vector_store_retry_count"] += r.get("retry_count", 0)
        metrics["failed_chunk_count"] += len(r.get("failed_chunks", []))
        metrics["error_count"] += sum(
            1 for k in ("error", "embedding_error", "store_error") if r.get(k)
        )
        metrics["vector_store_error_count"] += bool(r.get("store_error"))
    return metrics


def _log_progress(run_id: str | None, results: list[dict]) -> None:
    """Log cumulative progress metrics to ``run_id`` with ``step=docs_completed``.

    Uses ``MlflowClient`` with an explicit run id rather than the fluent API:
    this runs in LangGraph's entrypoint thread, which has no active run.
    Metric logging failures never fail the pipeline.
    """
    if not run_id:
        return
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Metric

        metrics = progress_metrics(results)
        step = metrics["docs_completed"]
        timestamp = int(time.time() * 1000)
        MlflowClient().log_batch(
            run_id,
            metrics=[Metric(k, v, timestamp, step) for k, v in metrics.items()],
        )
    except Exception as e:
        logger.warning("Failed to log progress metrics: %s", e)


@entrypoint()
def document_workflow(
    input_data: list[dict],
    config: RunnableConfig = {},
) -> list[dict]:
    """
    Run the full convert->chunk->embed->store pipeline for each document in parallel.

    Parallelism is capped by ``config["configurable"]["max_concurrency"]`` (default:
    CPU count). Documents are processed in batches; each batch runs convert, chunk,
    embed, and store for up to that many documents before the next batch starts.

    When ``config["configurable"]["mlflow_run_id"]`` is set, cumulative progress
    metrics (see ``progress_metrics``) are logged to that run after every batch.

    Parameters
    ----------
    input_data : list[dict]
        List of dicts, each with ``file_path`` (or legacy ``pdf_path``) and ``config``.
    config : RunnableConfig
        Run config; ``configurable.max_concurrency`` limits parallel document pipelines,
        ``configurable.mlflow_run_id`` enables per-batch progress metrics.

    Returns
    -------
    list[dict]
        One result dict per input, with "chunks", "embeddings", "stored_count",
        "store_error", "path", "error", and "config".
    """
    max_conc = _max_concurrency_from_config(config)
    run_id = (config or {}).get("configurable", {}).get("mlflow_run_id")
    results: list[dict] = []
    logger.info(" -> Running document workflow with max concurrency: %d", max_conc)

    for start in range(0, len(input_data), max_conc):
        batch = input_data[start : start + max_conc]
        converted = [
            convert_doc(
                {
                    "file_path": i.get("file_path") or i.get("pdf_path"),
                    "config": i.get("config"),
                }
            )
            for i in batch
        ]
        chunked = [chunk_doc({"converted": t}) for t in converted]
        embedded = [embed_doc({"chunked": t}) for t in chunked]
        stored = [store_doc({"embedded": t}) for t in embedded]
        results.extend(t.result() for t in stored)
        _log_progress(run_id, results)

    if not input_data:
        _log_progress(run_id, results)

    logger.info(" -> Document workflow completed with %d results", len(results))
    return results
