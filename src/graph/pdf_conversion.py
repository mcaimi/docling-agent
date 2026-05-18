#!/usr/bin/env python
"""Convert PDF files to text records using docling (standard or remote VLM pipeline)."""

import json
import os
from typing import Any, Literal, cast

# import the docling modules
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
    raise ImportError(f"Failed to import required modules: {e}") from e

_standard_converter: DocumentConverter | None = None
_vlm_converter_cache: tuple[str, DocumentConverter] | None = None


# instantiate the Docling DocumentConverter for the standard pipeline
def _get_standard_converter() -> DocumentConverter:
    global _standard_converter
    if _standard_converter is None:
        _standard_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
        )
    return _standard_converter


# create a cache key for the VLM converter
# uses the URL, preset, headers, and params to create a unique key
# this is used to cache the VLM converter
def _vlm_cache_key(
    url: str,
    preset: str,
    headers: dict[str, str],
    params: dict[str, Any],
) -> str:
    return json.dumps(
        {"url": url, "preset": preset, "headers": headers, "params": params},
        sort_keys=True,
    )


# create a new VLM converter
# if not already cached, put the converter in the cache
# and return the converter
def _get_vlm_converter(
    url: str,
    preset: str,
    headers: dict[str, str],
    params: dict[str, Any],
) -> DocumentConverter:
    global _vlm_converter_cache
    key = _vlm_cache_key(url, preset, headers, params)
    if _vlm_converter_cache is not None and _vlm_converter_cache[0] == key:
        return _vlm_converter_cache[1]

    vlm_options = VlmConvertOptions.from_preset(
        preset,
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=AnyUrl(url),
            headers=headers,
            params=params,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,
        vlm_options=vlm_options,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        },
    )
    _vlm_converter_cache = (key, converter)
    return converter


# enrich the conversion result from the docling document converter
# into a record with the content and metadata
def _conversion_result_to_record(
    pdf_path: str,
    conv: ConversionResult,
    conversion_pipeline: Literal["standard", "vlm"],
) -> dict[str, Any]:
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


# main function to convert the PDF paths to documents
# uses the pipeline to determine the converter to use
# returns a list of documents
def convert_pdf_paths_to_documents(
    pdf_paths: list[str],
    *,
    pipeline: Literal["standard", "vlm"] = "standard",
    vlm_api_url: str | None = None,
    vlm_api_headers: dict[str, str] | None = None,
    vlm_api_params: dict[str, Any] | None = None,
    vlm_preset: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert each PDF path to a record ``{"content": str, "metadata": dict}``.

    Use ``pipeline="standard"`` for the default docling PDF pipeline, or
    ``pipeline="vlm"`` for ``VlmPipeline`` with a remote OpenAI-compatible
    chat endpoint. When ``pipeline`` is omitted, ``DOCLING_PDF_PIPELINE`` is
    used (default ``standard``).

    For VLM mode, set ``DOCLING_VLM_API_URL_ENV`` or pass ``vlm_api_url``. Optional
    ``DOCLING_VLM_API_HEADERS_ENV`` / ``DOCLING_VLM_API_PARAMS_ENV`` (JSON objects) or
    the corresponding keyword arguments; ``DOCLING_VLM_PRESET_ENV`` (default
    ``smoldocling``) or ``vlm_preset``.

    Failures for individual files are logged to stdout; successful paths are included.
    """
    mode = pipeline
    print(f"Pipeline: {mode}")

    if mode == "vlm":
        print(f"VLM API URL: {vlm_api_url}")
        url = vlm_api_url.strip()
        preset = vlm_preset.strip()
        headers = vlm_api_headers
        params = vlm_api_params
        if not url:
            raise ValueError("VLM pipeline requires vlm_api_url=...")
        converter = _get_vlm_converter(url, preset, headers, params)
    else:
        converter = _get_standard_converter()

    documents: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        try:
            result = converter.convert(pdf_path)
            documents.append(_conversion_result_to_record(pdf_path, result, mode))
            print(f"Converted ({mode}): {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Error converting {pdf_path}: {str(e)}")

    return documents
