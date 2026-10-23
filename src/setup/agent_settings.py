#!/usr/bin/env python
"""
Resolved agent configuration: ``.env`` overrides defaults from ``src.config``.

Load order: ``python-dotenv`` reads the project ``.env`` into ``os.environ``;
each setting prefers a non-empty environment value, otherwise the value from
``src.config``.
"""

import os
from pathlib import Path
from typing import Any

# import required modules
try:
    from dotenv import load_dotenv
    from ..utils import _parse_json_object_str
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
def load_environment_variables(envfile: str) -> dict[str, Any]:
    e_file = Path(envfile)
    if not e_file.exists():
        raise FileNotFoundError(f"Environment file {e_file} not found")
    load_dotenv(e_file)  # load the environment variables
    return _get_settings_from_environment()


# get the settings
def _get_settings_from_environment():
    settings = {
        "llm_model": _str_setting("LLM_MODEL", "gpt-4o-mini"),
        "base_url": _str_setting("BASE_URL", "https://api.openai.com/v1"),
        "api_key": _str_setting("API_KEY", "your-api-key-here"),
        "folder_path": _str_setting("FOLDER_PATH", "./pdfs"),
        "embedding_model": _str_setting("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "embedding_dimensions": _int_setting("EMBEDDING_DIMENSIONS", 1536),
        "embedding_base_url": _str_setting(
            "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
        ),
        "embedding_api_key": _str_setting("EMBEDDING_API_KEY", "your-api-key-here"),
        "conversion_pipeline": _str_setting("CONVERSION_PIPELINE", "standard"),
        "chunking_strategy": _str_setting("CHUNKING_STRATEGY", "fixed"),
        "chunk_size": _int_setting("CHUNK_SIZE", 1000),
        "chunk_overlap": _int_setting("CHUNK_OVERLAP", 100),
        "vlm_api_url": _str_setting("VLM_API_URL", "https://api.openai.com/v1"),
        "vlm_api_headers": _parse_json_object_str(
            _str_setting("VLM_API_HEADERS", "{}")
        ),
        "vlm_api_params": _parse_json_object_str(_str_setting("VLM_API_PARAMS", "{}")),
        "vlm_preset": _str_setting("VLM_PRESET", "smoldocling"),
        "max_concurrency": _int_setting("MAX_CONCURRENCY", os.cpu_count()),
        "enable_tracing": _int_setting("ENABLE_TRACING", 0),
        "mlflow_tracking_uri": _str_setting("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "mlflow_experiment_name": _str_setting("MLFLOW_EXPERIMENT_NAME", "Docling Agent"),
        "mlflow_session_id": _strip_or_none("MLFLOW_SESSION_ID"),
        "vector_store_backend": _str_setting("VECTOR_STORE_BACKEND", "none"),
        "vector_store_collection": _str_setting(
            "VECTOR_STORE_COLLECTION", "docling_chunks"
        ),
        "chroma_host": _strip_or_none("CHROMA_HOST"),
        "chroma_port": _int_setting("CHROMA_PORT", 8000),
        "chroma_persist_directory": _strip_or_none("CHROMA_PERSIST_DIRECTORY"),
        "milvus_uri": _str_setting("MILVUS_URI", "http://localhost:19530"),
        "milvus_token": _strip_or_none("MILVUS_TOKEN"),
        "milvus_db_name": _str_setting("MILVUS_DB_NAME", "default"),
        "vector_store_batch_size": _int_setting("VECTOR_STORE_BATCH_SIZE", 200),
        "vector_store_write_mode": _str_setting("VECTOR_STORE_WRITE_MODE", "insert"),
        "vector_store_max_batch_workers": _int_setting("VECTOR_STORE_MAX_BATCH_WORKERS", 4),
        "vector_store_retry_attempts": _int_setting("VECTOR_STORE_RETRY_ATTEMPTS", 3),
        "vector_store_retry_backoff_multiplier": _int_setting(
            "VECTOR_STORE_RETRY_BACKOFF_MULTIPLIER", 1
        ),
        "vector_store_circuit_breaker_threshold": _int_setting(
            "VECTOR_STORE_CIRCUIT_BREAKER_THRESHOLD", 5
        ),
        "vector_store_circuit_breaker_timeout": _int_setting(
            "VECTOR_STORE_CIRCUIT_BREAKER_TIMEOUT", 60
        ),
        "milvus_index_type": _str_setting("MILVUS_INDEX_TYPE", "HNSW"),
        "milvus_metric_type": _str_setting("MILVUS_METRIC_TYPE", "COSINE"),
        "milvus_index_params": _parse_json_object_str(
            _str_setting("MILVUS_INDEX_PARAMS", "{}")
        ),
        "milvus_consistency_level": _str_setting("MILVUS_CONSISTENCY_LEVEL", "Bounded"),
        "milvus_partition_name": _strip_or_none("MILVUS_PARTITION_NAME"),
        "chroma_distance_function": _str_setting("CHROMA_DISTANCE_FUNCTION", "cosine"),
        "chroma_collection_metadata": _parse_json_object_str(
            _str_setting("CHROMA_COLLECTION_METADATA", "{}")
        ),
        "chroma_allow_reset": _int_setting("CHROMA_ALLOW_RESET", 0),
    }
    return settings


# export the settings
__all__ = [
    "load_environment_variables",
]
