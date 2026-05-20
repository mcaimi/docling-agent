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
    from ..utils.mlflow_tracing import session_context
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

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

    return DocumentConverter(allowed_formats=[InputFormat.PDF])


# convert docling data to dictionary
def _result_to_record(
    pdf_path: str,
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
        title = os.path.splitext(os.path.basename(pdf_path))[0]

    metadata: dict[str, Any] = {
        "file_path": pdf_path,
        "file_name": os.path.basename(pdf_path),
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
    """Convert a single PDF with a dedicated DocumentConverter instance.

    Parameters
    ----------
    item : {"pdf_path": str, "config": dict}
        Contains the file path and agent config.

    Returns
    -------
    dict
        {"doc": record|None, "path": str, "error": str|None, "config": dict}
    """
    config = item["config"]
    pdf_path = item["pdf_path"]
    pipeline = config.get("conversion_pipeline")
    session_id = config.get("mlflow_session_id") if enable_tracing else None

    # displayinfo
    print(f" -> conversion task in path {os.path.basename(pdf_path)}, using pipeline mode: {pipeline}")

    with session_context(session_id):
        try:
            converter = _new_converter(
                pipeline=pipeline,
                vlm_api_url=config.get("vlm_api_url"),
                vlm_api_headers=config.get("vlm_api_headers"),
                vlm_api_params=config.get("vlm_api_params"),
                vlm_preset=config.get("vlm_preset"),
            )

            result = converter.convert(pdf_path)
            doc = _result_to_record(pdf_path, result, pipeline)

            if enable_tracing:
                with _ml.start_span("convert_pdf"):
                    _ml.set_tag("pipeline", pipeline)
                    _ml.set_tag("file", os.path.basename(pdf_path))
                    _ml.set_tag("page_count", result.input.page_count)

            return {"doc": doc, "path": pdf_path, "error": None, "config": config}
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("convert_pdf"):
                    _ml.set_tag("error", str(e))
            print(f"Error converting {pdf_path}: {e}")
            return {"doc": None, "path": pdf_path, "error": str(e), "config": config}


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

        if strategy == "hybrid":
            chunks = chunk_document_records_hybrid([doc])
        else:
            chunks = chunk_document_records(
                [doc],
                config.get("chunk_size", 1000),
                config.get("chunk_overlap", 100),
            )

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
        {"chunks": [...], "embeddings": [...], "path": str, "error": str|None, "config": dict}
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
                "config": config,
            }

        chunks = result["chunks"]
        texts = [c["content"] for c in chunks]

        # display info
        print(f" -> Embedding {len(texts)} chunks...")

        try:
            vectors = embed_texts(
                texts,
                config.get("embedding_model"),
                config.get("embedding_base_url"),
                config.get("embedding_api_key", ""),
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
        except Exception as e:
            if enable_tracing:
                with _ml.start_span("embed_doc"):
                    _ml.set_tag("error", str(e))
            raise

        print(f" -> Produced {len(vectors)} embeddings.")
        return {
            "chunks": chunks,
            "embeddings": vectors,
            "path": result["path"],
            "error": None,
            "config": config,
        }


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
    Run the full convert->chunk->embed pipeline for each document in parallel.

    Parallelism is capped by ``config["configurable"]["max_concurrency"]`` (default:
    CPU count). Documents are processed in batches; each batch runs convert, chunk,
    and embed for up to that many PDFs before the next batch starts.

    Parameters
    ----------
    input_data : list[dict]
        List of dicts, each with {"pdf_path": str, "config": dict}.
    config : RunnableConfig
        Run config; ``configurable.max_concurrency`` limits parallel PDF pipelines.

    Returns
    -------
    list[dict]
        One result dict per input, with "chunks", "embeddings", "path",
        "error", and "config".
    """
    max_conc = _max_concurrency_from_config(config)
    results: list[dict] = []
    print(f" -> Running document workflow with max concurrency: {max_conc}")

    for start in range(0, len(input_data), max_conc):
        batch = input_data[start : start + max_conc]
        converted = [
            convert_doc({"pdf_path": i.get("pdf_path"), "config": i.get("config")})
            for i in batch
        ]
        chunked = [chunk_doc({"converted": t}) for t in converted]
        embedded = [embed_doc({"chunked": t}) for t in chunked]
        results.extend(t.result() for t in embedded)

    print(f" -> Document workflow completed with {len(results)} results")
    return results
