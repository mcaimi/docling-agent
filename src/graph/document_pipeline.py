#!/usr/bin/env python
"""Per-document parallel processing tasks using langgraph.func API."""

import os
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
    from ..utils.mlflow_tracing import session_context
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

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
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

    # displayinfo
    print(
        f" -> conversion task in path {os.path.basename(file_path)}, "
        f"using pipeline mode: {pipeline}"
        + (
            f" (VLM skipped for non-PDF; requested {requested_pipeline})"
            if pipeline != requested_pipeline
            else ""
        )
    )

    with session_context(session_id):
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

            if enable_tracing:
                with _ml.start_span("convert_pdf"):
                    _ml.set_tag("pipeline", pipeline)
                    _ml.set_tag("file", os.path.basename(file_path))
                    _ml.set_tag("file_format", os.path.splitext(file_path)[1].lower())
                    _ml.set_tag("page_count", result.input.page_count)

            return {"doc": doc, "path": file_path, "error": None, "config": config}
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("convert_pdf"):
                    _ml.set_tag("error", str(e))
            print(f"Error converting {file_path}: {e}")
            return {"doc": None, "path": file_path, "error": str(e), "config": config}


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

    with session_context(session_id):
        if result.get("error") or result["doc"] is None:
            return {
                "chunks": [],
                "path": result["path"],
                "error": result["error"],
                "config": config,
            }

        strategy = config.get("chunking_strategy")
        doc = result["doc"]

        try:
            if strategy == "hybrid":
                chunks = chunk_document_records_hybrid([doc])
            else:
                chunks = chunk_document_records(
                    [doc],
                    config.get("chunk_size", 1000),
                    config.get("chunk_overlap", 100),
                )
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("chunk_doc"):
                    _ml.set_tag("error", str(e))
            print(f"Error chunking {result['path']}: {e}")
            return {
                "chunks": [],
                "path": result["path"],
                "error": str(e),
                "config": config,
            }

        if enable_tracing:
            with _ml.start_span("chunk_doc"):
                _ml.set_tag("chunking_strategy", strategy)
                _ml.set_tag("chunk_count", len(chunks))

        print(
            f" -> Chunking completed for path {result.get('path')}: produced {len(chunks)} chunks"
        )
        return {"chunks": chunks, "path": result["path"], "error": None, "config": config}


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

    with session_context(session_id):
        if result.get("error") or not result["chunks"]:
            return {
                "embeddings": [],
                "path": result["path"],
                "error": result["error"],
                "embedding_error": None,
                "config": config,
            }

        chunks = result["chunks"]
        texts = [c["content"] for c in chunks]

        # display info
        print(f" -> Embedding {len(texts)} chunks...")

        try:
            vectors, embed_errors = embed_texts(
                texts,
                config.get("embedding_model"),
                config.get("embedding_base_url"),
                config.get("embedding_api_key", ""),
                config.get("embedding_dimensions"),
            )
            failed_count = sum(1 for v in vectors if v is None)
            embedding_error = (
                f"{failed_count}/{len(vectors)} chunks failed to embed: "
                + "; ".join(embed_errors)
                if failed_count
                else None
            )
            if enable_tracing:
                with _ml.start_span("embed_doc"):
                    _ml.set_tag(
                        "embedding_model", config.get("embedding_model", "unknown")
                    )
                    _ml.set_tag("chunk_count", len(texts))
                    _ml.set_tag("embedding_count", len(vectors))
                    _ml.set_tag(
                        "non_null_count", sum(1 for v in vectors if v is not None)
                    )
                    if embedding_error:
                        _ml.set_tag("embedding_error", embedding_error)
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("embed_doc"):
                    _ml.set_tag("error", str(e))
            raise

        print(f" -> Produced {len(vectors)} embeddings.")
        if embedding_error:
            print(f" -> {embedding_error}")
        return {
            "chunks": chunks,
            "embeddings": vectors,
            "path": result["path"],
            "error": None,
            "embedding_error": embedding_error,
            "config": config,
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
    backend = (config.get("vector_store_backend") or "none").strip().lower()

    print(f" -> Vector store backend: {backend}")

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

    with session_context(session_id):
        if (
            backend in ("", "none")
            or result.get("error")
            or not result.get("chunks")
        ):
            return base

        print(
            f" -> Storing embeddings for {os.path.basename(result['path'])} "
            f"({backend})..."
        )
        try:
            vector_store = get_vector_store(config)
            if vector_store is None:
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
            )
            stored_count = store_result["stored_count"]
            skipped = store_result["skipped_count"]
            upserted = store_result.get("upserted_count", 0)
            retry_count = store_result.get("retry_count", 0)
            failed_chunks = store_result.get("failed_chunks", [])

            if enable_tracing:
                with _ml.start_span("store_doc"):
                    _ml.set_tag("backend", backend)
                    _ml.set_tag("file", os.path.basename(result["path"]))
                    _ml.set_tag("write_mode", write_mode)
                    _ml.set_tag("stored_count", stored_count)
                    _ml.set_tag("skipped_count", skipped)
                    _ml.set_tag("upserted_count", upserted)
                    _ml.set_tag("retry_count", retry_count)
                    _ml.set_tag("failed_count", len(failed_chunks))

            print(
                f" -> Stored {stored_count} vectors ({skipped} skipped, "
                f"{upserted} upserted, {retry_count} retries, {len(failed_chunks)} failed)."
            )
            return {
                **base,
                "stored_count": stored_count,
                "upserted_count": upserted,
                "retry_count": retry_count,
                "failed_chunks": failed_chunks,
            }
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("store_doc"):
                    _ml.set_tag("backend", backend)
                    _ml.set_tag("error", str(e))
            print(f"Error storing vectors for {result['path']}: {e}")
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

    Parameters
    ----------
    input_data : list[dict]
        List of dicts, each with ``file_path`` (or legacy ``pdf_path``) and ``config``.
    config : RunnableConfig
        Run config; ``configurable.max_concurrency`` limits parallel document pipelines.

    Returns
    -------
    list[dict]
        One result dict per input, with "chunks", "embeddings", "stored_count",
        "store_error", "path", "error", and "config".
    """
    max_conc = _max_concurrency_from_config(config)
    results: list[dict] = []
    print(f" -> Running document workflow with max concurrency: {max_conc}")

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

    print(f" -> Document workflow completed with {len(results)} results")
    return results
