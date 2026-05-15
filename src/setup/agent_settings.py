#!/usr/bin/env python
"""
Resolved agent configuration: ``.env`` overrides defaults from ``src.config``.

Load order: ``python-dotenv`` reads the project ``.env`` into ``os.environ``;
each setting prefers a non-empty environment value, otherwise the value from
``src.config``.
"""

import os
from pathlib import Path

# import dotenv
try:
    from dotenv import load_dotenv
    from . import config as _defaults
except ImportError:
    raise ImportError("python-dotenv is required to use this module")


# loads an env variable and strips it of whitespace
# None if not set
def _strip_or_none(key: str) -> str | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


# returns a string setting from the environment
# default if not set
def _str_setting(key: str, default: str) -> str:
    return _strip_or_none(key) or default


# same as _str_setting but for ints
def _int_setting(key: str, default: int) -> int:
    raw = _strip_or_none(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# load the environment variables
try:
    # dotenv file location
    dotenv_file = Path(__file__).parent.parent / ".env"
    print(f"Loading environment variables from {dotenv_file}")
    load_dotenv(dotenv_file)
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# settings used throughout the agent.
EMBEDDING_MODEL = _str_setting("EMBEDDING_MODEL", _defaults.EMBEDDING_MODEL)
EMBEDDING_DIMENSIONS = _int_setting(
    "EMBEDDING_DIMENSIONS", _defaults.EMBEDDING_DIMENSIONS
)
CHUNK_SIZE = _int_setting("CHUNK_SIZE", _defaults.CHUNK_SIZE)
CHUNK_OVERLAP = _int_setting("CHUNK_OVERLAP", _defaults.CHUNK_OVERLAP)
DOCLING_CHUNKING_STRATEGY = _str_setting(
    "DOCLING_CHUNKING_STRATEGY", _defaults.DOCLING_CHUNKING_STRATEGY
)
PDF_FOLDER_PATH = _str_setting("PDF_FOLDER_PATH", _defaults.PDF_FOLDER_PATH)

# Optional embedding HTTP settings: env wins; config supplies Python-side default.
EMBEDDING_BASE_URL = (
    _strip_or_none("EMBEDDING_BASE_URL") or _defaults.EMBEDDING_BASE_URL
)
EMBEDDING_API_KEY = _strip_or_none("EMBEDDING_API_KEY") or _defaults.EMBEDDING_API_KEY
OPENAI_API_KEY = _strip_or_none("OPENAI_API_KEY") or _defaults.OPENAI_API_KEY

# Docling PDF pipeline (used when convert_pdf_paths_to_documents(pipeline=None))
DOCLING_PDF_PIPELINE_ENV = _str_setting(
    "DOCLING_PDF_PIPELINE_ENV", _defaults.DOCLING_PDF_PIPELINE_ENV
)
DOCLING_PDF_PIPELINE_DEFAULT = _str_setting(
    "DOCLING_PDF_PIPELINE_DEFAULT", _defaults.DOCLING_PDF_PIPELINE_DEFAULT
)

# Remote VLM (OpenAI-compatible chat completions URL)
DOCLING_VLM_API_URL_ENV = _str_setting(
    "DOCLING_VLM_API_URL_ENV", _defaults.DOCLING_VLM_API_URL_ENV
)
DOCLING_VLM_API_HEADERS_ENV = _str_setting(
    "DOCLING_VLM_API_HEADERS_ENV", _defaults.DOCLING_VLM_API_HEADERS_ENV
)
DOCLING_VLM_API_PARAMS_ENV = _str_setting(
    "DOCLING_VLM_API_PARAMS_ENV", _defaults.DOCLING_VLM_API_PARAMS_ENV
)
DOCLING_VLM_PRESET_ENV = _str_setting(
    "DOCLING_VLM_PRESET_ENV", _defaults.DOCLING_VLM_PRESET_ENV
)
DOCLING_VLM_PRESET_DEFAULT = _str_setting(
    "DOCLING_VLM_PRESET_DEFAULT", _defaults.DOCLING_VLM_PRESET_DEFAULT
)

# export the settings
__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DOCLING_CHUNKING_STRATEGY",
    "PDF_FOLDER_PATH",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
    "DOCLING_PDF_PIPELINE_ENV",
    "DOCLING_PDF_PIPELINE_DEFAULT",
    "DOCLING_VLM_API_URL_ENV",
    "DOCLING_VLM_API_HEADERS_ENV",
    "DOCLING_VLM_API_PARAMS_ENV",
    "DOCLING_VLM_PRESET_ENV",
    "DOCLING_VLM_PRESET_DEFAULT",
]

