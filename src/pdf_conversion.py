"""Convert PDF files to text records using docling (standard or remote VLM pipeline)."""

import json
import os

from src import agent_settings as _agent_settings  # noqa: F401 — load .env before os.getenv below
from typing import Any, Literal, cast

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from pydantic import AnyUrl

from src.config import (
    DOCLING_PDF_PIPELINE_DEFAULT,
    DOCLING_PDF_PIPELINE_ENV,
    DOCLING_VLM_API_HEADERS_ENV,
    DOCLING_VLM_API_PARAMS_ENV,
    DOCLING_VLM_API_URL_ENV,
    DOCLING_VLM_PRESET_DEFAULT,
    DOCLING_VLM_PRESET_ENV,
)

_standard_converter: DocumentConverter | None = None
_vlm_converter_cache: tuple[str, DocumentConverter] | None = None


def _parse_json_object_env(var_name: str) -> dict[str, Any]:
    raw = os.getenv(var_name)
    if not raw or not raw.strip():
        return {}
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {var_name}: {e}") from e
    if not isinstance(out, dict):
        raise ValueError(f"{var_name} must be a JSON object, got {type(out).__name__}")
    return cast(dict[str, Any], out)


def _resolved_pipeline(pipeline: Literal["standard", "vlm"] | None) -> Literal["standard", "vlm"]:
    if pipeline is not None:
        return pipeline
    raw = (os.getenv(DOCLING_PDF_PIPELINE_ENV) or DOCLING_PDF_PIPELINE_DEFAULT).strip().lower()
    if raw in ("standard", "vlm"):
        return cast(Literal["standard", "vlm"], raw)
    print(f"Unknown {DOCLING_PDF_PIPELINE_ENV}={raw!r}, using {DOCLING_PDF_PIPELINE_DEFAULT}")
    return "standard"


def _get_standard_converter() -> DocumentConverter:
    global _standard_converter
    if _standard_converter is None:
        _standard_converter = DocumentConverter()
    return _standard_converter


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
    return {"content": text_content, "metadata": metadata}


def convert_pdf_paths_to_documents(
    pdf_paths: list[str],
    *,
    pipeline: Literal["standard", "vlm"] | None = None,
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

    For VLM mode, set ``DOCLING_VLM_API_URL`` or pass ``vlm_api_url``. Optional
    ``DOCLING_VLM_API_HEADERS`` / ``DOCLING_VLM_API_PARAMS`` (JSON objects) or
    the corresponding keyword arguments; ``DOCLING_VLM_PRESET`` (default
    ``smoldocling``) or ``vlm_preset``.

    Failures for individual files are logged to stdout; successful paths are included.
    """
    mode = _resolved_pipeline(pipeline)

    headers_env = _parse_json_object_env(DOCLING_VLM_API_HEADERS_ENV)
    headers_merged: dict[str, str] = {str(k): str(v) for k, v in headers_env.items()}
    if vlm_api_headers is not None:
        headers_merged = {str(k): str(v) for k, v in vlm_api_headers.items()}

    params_merged: dict[str, Any] = dict(_parse_json_object_env(DOCLING_VLM_API_PARAMS_ENV))
    if vlm_api_params is not None:
        params_merged = dict(vlm_api_params)

    preset = (vlm_preset or os.getenv(DOCLING_VLM_PRESET_ENV) or DOCLING_VLM_PRESET_DEFAULT).strip()

    if mode == "vlm":
        url = (vlm_api_url or os.getenv(DOCLING_VLM_API_URL_ENV) or "").strip()
        if not url:
            raise ValueError(
                f"VLM pipeline requires {DOCLING_VLM_API_URL_ENV} or vlm_api_url=..."
            )
        converter = _get_vlm_converter(url, preset, headers_merged, params_merged)
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
