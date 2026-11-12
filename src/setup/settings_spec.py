#!/usr/bin/env python
"""
Declarative specification of every agent setting.

Single source of truth for:
- environment loading (``agent_settings._get_settings_from_environment``)
- secret handling: ``sensitive`` settings are redacted from MLflow params and,
  in a launcher, rendered into a Secret instead of plain env/ConfigMap
- the settings contract (``settings_contract()``): a JSON document that
  launchers such as docling-scheduler vendor instead of importing this code.
  ``python -m src.docling_agent --print-settings-schema`` prints it, and
  ``settings-schema.json`` at the repo root is the committed copy.

The spec module has no third-party imports.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

# ``bool`` values are stored as int 0/1 -- the historical representation of
# flags such as ENABLE_TRACING -- and accept 1/0, true/false, yes/no, on/off.
# ``json`` values must be JSON objects.
SettingType = Literal["str", "int", "bool", "json"]


@dataclass(frozen=True)
class SettingSpec:
    key: str
    env: str
    type: SettingType
    # Used when the env var is unset, blank, or (for int/bool) unparseable.
    # A callable is invoked at load time (e.g. CPU count). ``None`` means
    # "unset / no override".
    default: Any | Callable[[], Any]
    group: str
    description: str = ""
    choices: tuple[str, ...] | None = None
    # Credential, or may embed one: never logged, never sent to the browser.
    sensitive: bool = False
    # Free text from an untrusted source (see prompts/extractor.py).
    untrusted: bool = False
    # Additional env vars consulted, in order, when ``env`` is unset.
    env_fallbacks: tuple[str, ...] = ()
    # UI hint: only relevant when another setting has one of these values.
    depends_on: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def resolve_default(self) -> Any:
        return self.default() if callable(self.default) else self.default


def _cpu_count() -> int | None:
    return os.cpu_count()


# Computed defaults by contract name: the contract names them instead of
# resolving them on the machine that generates it.
_DYNAMIC_DEFAULTS: dict[Callable[[], Any], str] = {_cpu_count: "cpu_count"}

# Bumped on incompatible changes to the contract's shape (not its contents).
CONTRACT_VERSION = 1


_VLM = {"conversion_pipeline": ("vlm",)}
_FIXED = {"chunking_strategy": ("fixed",)}
_TRACING = {"enable_tracing": ("1",)}
_CHROMA = {"vector_store_backend": ("chroma",)}
_MILVUS = {"vector_store_backend": ("milvus",)}
_STORE = {"vector_store_backend": ("chroma", "milvus")}

_S = SettingSpec

SETTINGS_SPEC: tuple[SettingSpec, ...] = (
    # --- free-text prompt ---
    _S("user_prompt", "USER_PROMPT", "str", "", "prompt",
       "Natural-language instructions; may override non-sensitive settings", untrusted=True),
    # --- LLM used for NLP extraction from user_prompt ---
    _S("llm_model", "LLM_MODEL", "str", "gpt-4o-mini", "llm",
       "Model that extracts config overrides from the user prompt"),
    _S("base_url", "BASE_URL", "str", "https://api.openai.com/v1", "llm",
       "OpenAI-compatible base URL of the extraction LLM"),
    _S("api_key", "API_KEY", "str", "your-api-key-here", "llm",
       "API key of the extraction LLM", sensitive=True),
    # --- source documents ---
    _S("folder_path", "FOLDER_PATH", "str", "./pdfs", "source",
       "Folder scanned for supported documents"),
    _S("shard_index", "SHARD_INDEX", "int", 0, "source",
       "Zero-based shard processed by this instance", env_fallbacks=("JOB_COMPLETION_INDEX",)),
    _S("shard_count", "SHARD_COUNT", "int", 1, "source",
       "Number of shards the sorted document list is split into"),
    # --- conversion ---
    _S("conversion_pipeline", "CONVERSION_PIPELINE", "str", "standard", "conversion",
       "Docling conversion pipeline (VLM applies to PDF only)", choices=("standard", "vlm")),
    _S("vlm_api_url", "VLM_API_URL", "str", "https://api.openai.com/v1", "conversion",
       "OpenAI-compatible endpoint of the remote VLM", depends_on=_VLM),
    _S("vlm_api_headers", "VLM_API_HEADERS", "json", {}, "conversion",
       "HTTP headers sent to the VLM endpoint (JSON object)", sensitive=True, depends_on=_VLM),
    _S("vlm_api_params", "VLM_API_PARAMS", "json", {}, "conversion",
       "Extra request params for the VLM endpoint (JSON object)", sensitive=True, depends_on=_VLM),
    _S("vlm_preset", "VLM_PRESET", "str", "smoldocling", "conversion",
       "Docling VLM preset", depends_on=_VLM),
    # --- chunking ---
    _S("chunking_strategy", "CHUNKING_STRATEGY", "str", "fixed", "chunking",
       "Chunking strategy", choices=("fixed", "hybrid")),
    _S("chunk_size", "CHUNK_SIZE", "int", 1000, "chunking",
       "Chunk size in characters", depends_on=_FIXED),
    _S("chunk_overlap", "CHUNK_OVERLAP", "int", 100, "chunking",
       "Chunk overlap in characters (must be < chunk size)", depends_on=_FIXED),
    # --- embeddings ---
    _S("embedding_model", "EMBEDDING_MODEL", "str", "text-embedding-ada-002", "embeddings",
       "Embedding model name"),
    # None means "use the model's native dimension" -- only set this for
    # models that accept a custom `dimensions` request param (e.g.
    # text-embedding-3-*); sending it to models that don't (e.g. the default
    # text-embedding-ada-002) is a request error.
    _S("embedding_dimensions", "EMBEDDING_DIMENSIONS", "int", None, "embeddings",
       "Custom output dimension (only for models that support it)"),
    _S("embedding_base_url", "EMBEDDING_BASE_URL", "str", "https://api.openai.com/v1", "embeddings",
       "OpenAI-compatible base URL for embeddings"),
    _S("embedding_api_key", "EMBEDDING_API_KEY", "str", "your-api-key-here", "embeddings",
       "API key of the embedding endpoint", sensitive=True),
    # --- runtime ---
    _S("max_concurrency", "MAX_CONCURRENCY", "int", _cpu_count, "runtime",
       "Documents processed in parallel per batch"),
    _S("log_level", "LOG_LEVEL", "str", "INFO", "runtime",
       "Log level", choices=("DEBUG", "INFO", "WARNING", "ERROR")),
    _S("log_format", "LOG_FORMAT", "str", "text", "runtime",
       "Log format (json = one object per line, for container log collection)",
       choices=("text", "json")),
    # --- MLflow tracing ---
    _S("enable_tracing", "ENABLE_TRACING", "bool", 0, "tracing",
       "Enable MLflow tracing"),
    _S("mlflow_tracking_uri", "MLFLOW_TRACKING_URI", "str", "http://localhost:5000", "tracing",
       "MLflow tracking server URI", depends_on=_TRACING),
    _S("mlflow_experiment_name", "MLFLOW_EXPERIMENT_NAME", "str", "Docling Agent", "tracing",
       "MLflow experiment name", depends_on=_TRACING),
    # Off by default: autolog records every LangGraph task's inputs/outputs,
    # including whole DoclingDocuments whose uint64 ``binary_hash`` overflows
    # the tracking server's int64 fields -- the HTTP backend then drops the
    # entire trace (verified: ~half the per-document traces were lost).
    _S("mlflow_langchain_autolog", "MLFLOW_LANGCHAIN_AUTOLOG", "bool", 0, "tracing",
       "Also autolog LangGraph task spans (large payloads; can drop traces on an MLflow server)",
       depends_on=_TRACING),
    _S("mlflow_session_id", "MLFLOW_SESSION_ID", "str", None, "tracing",
       "Trace session id (default: parent run id, else this run's id)", depends_on=_TRACING),
    _S("mlflow_parent_run_id", "MLFLOW_PARENT_RUN_ID", "str", None, "tracing",
       "Log this run as a nested child of the given MLflow run", depends_on=_TRACING),
    # --- scheduler metadata (set by the scheduler, not by users) ---
    _S("scheduler_run_id", "SCHEDULER_RUN_ID", "str", None, "scheduler",
       "Logical run id assigned by the scheduler"),
    _S("scheduler_executor", "SCHEDULER_EXECUTOR", "str", None, "scheduler",
       "Executor that launched this instance", choices=("podman", "kubernetes")),
    # --- vector store ---
    _S("vector_store_backend", "VECTOR_STORE_BACKEND", "str", "none", "vector_store",
       "Vector store backend", choices=("none", "chroma", "milvus")),
    _S("vector_store_collection", "VECTOR_STORE_COLLECTION", "str", "docling_chunks", "vector_store",
       "Collection name", depends_on=_STORE),
    _S("vector_store_write_mode", "VECTOR_STORE_WRITE_MODE", "str", "insert", "vector_store",
       "Write mode", choices=("insert", "upsert", "skip_existing"), depends_on=_STORE),
    _S("vector_store_batch_size", "VECTOR_STORE_BATCH_SIZE", "int", 200, "vector_store",
       "Vectors per write batch", depends_on=_STORE),
    _S("vector_store_max_batch_workers", "VECTOR_STORE_MAX_BATCH_WORKERS", "int", 4, "vector_store",
       "Parallel write workers per document", depends_on=_STORE),
    _S("vector_store_retry_attempts", "VECTOR_STORE_RETRY_ATTEMPTS", "int", 3, "vector_store",
       "Retry attempts for transient write failures", depends_on=_STORE),
    _S("vector_store_retry_backoff_multiplier", "VECTOR_STORE_RETRY_BACKOFF_MULTIPLIER", "int", 1,
       "vector_store", "Exponential backoff multiplier", depends_on=_STORE),
    _S("vector_store_circuit_breaker_threshold", "VECTOR_STORE_CIRCUIT_BREAKER_THRESHOLD", "int", 5,
       "vector_store", "Consecutive failures before the circuit opens", depends_on=_STORE),
    _S("vector_store_circuit_breaker_timeout", "VECTOR_STORE_CIRCUIT_BREAKER_TIMEOUT", "int", 60,
       "vector_store", "Seconds before a half-open recovery attempt", depends_on=_STORE),
    _S("chroma_host", "CHROMA_HOST", "str", None, "vector_store",
       "Chroma server host (unset = embedded client)", depends_on=_CHROMA),
    _S("chroma_port", "CHROMA_PORT", "int", 8000, "vector_store",
       "Chroma server port", depends_on=_CHROMA),
    _S("chroma_persist_directory", "CHROMA_PERSIST_DIRECTORY", "str", None, "vector_store",
       "Persistence directory of the embedded Chroma client", depends_on=_CHROMA),
    _S("chroma_distance_function", "CHROMA_DISTANCE_FUNCTION", "str", "cosine", "vector_store",
       "Chroma distance function", choices=("cosine", "l2", "ip"), depends_on=_CHROMA),
    _S("chroma_collection_metadata", "CHROMA_COLLECTION_METADATA", "json", {}, "vector_store",
       "Chroma collection metadata (JSON object)", depends_on=_CHROMA),
    _S("chroma_allow_reset", "CHROMA_ALLOW_RESET", "bool", 0, "vector_store",
       "Allow collection reset (testing only)", depends_on=_CHROMA),
    _S("milvus_uri", "MILVUS_URI", "str", "http://localhost:19530", "vector_store",
       "Milvus URI", depends_on=_MILVUS),
    _S("milvus_token", "MILVUS_TOKEN", "str", None, "vector_store",
       "Milvus auth token", sensitive=True, depends_on=_MILVUS),
    _S("milvus_db_name", "MILVUS_DB_NAME", "str", "default", "vector_store",
       "Milvus database", depends_on=_MILVUS),
    _S("milvus_index_type", "MILVUS_INDEX_TYPE", "str", "HNSW", "vector_store",
       "Milvus index type", choices=("HNSW", "IVF_FLAT", "FLAT", "AUTOINDEX"), depends_on=_MILVUS),
    _S("milvus_metric_type", "MILVUS_METRIC_TYPE", "str", "COSINE", "vector_store",
       "Milvus metric type", choices=("COSINE", "L2", "IP"), depends_on=_MILVUS),
    _S("milvus_index_params", "MILVUS_INDEX_PARAMS", "json", {}, "vector_store",
       "Milvus index params (JSON object)", depends_on=_MILVUS),
    _S("milvus_consistency_level", "MILVUS_CONSISTENCY_LEVEL", "str", "Bounded", "vector_store",
       "Milvus consistency level", choices=("Strong", "Bounded", "Session", "Eventually"),
       depends_on=_MILVUS),
    _S("milvus_partition_name", "MILVUS_PARTITION_NAME", "str", None, "vector_store",
       "Milvus partition", depends_on=_MILVUS),
)

SETTINGS_BY_KEY: dict[str, SettingSpec] = {s.key: s for s in SETTINGS_SPEC}

# Settings whose values are credentials or may embed them.
SENSITIVE_SETTING_KEYS: frozenset[str] = frozenset(s.key for s in SETTINGS_SPEC if s.sensitive)


def settings_contract(document_globs: Sequence[str]) -> dict[str, Any]:
    """JSON-serialisable contract between the agent and its launchers.

    Defaults of sensitive settings are never included; computed defaults are
    named in ``dynamic_default`` (``default`` is then ``None``).
    """
    settings = []
    for s in SETTINGS_SPEC:
        dynamic = _DYNAMIC_DEFAULTS.get(s.default) if callable(s.default) else None
        if callable(s.default) and dynamic is None:
            raise ValueError(f"{s.key}: callable default has no contract name")
        settings.append(
            {
                "key": s.key,
                "env": s.env,
                "type": s.type,
                "default": None if s.sensitive or dynamic else s.default,
                "dynamic_default": dynamic,
                "group": s.group,
                "description": s.description,
                "choices": list(s.choices) if s.choices else None,
                "sensitive": s.sensitive,
                "untrusted": s.untrusted,
                "env_fallbacks": list(s.env_fallbacks),
                "depends_on": {k: list(v) for k, v in s.depends_on.items()},
            }
        )
    return {
        "schema_version": CONTRACT_VERSION,
        "document_globs": list(document_globs),
        "settings": settings,
    }


__all__ = [
    "SettingSpec",
    "SETTINGS_SPEC",
    "SETTINGS_BY_KEY",
    "SENSITIVE_SETTING_KEYS",
    "CONTRACT_VERSION",
    "settings_contract",
]
