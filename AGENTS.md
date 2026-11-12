# Docling Document Processing Agent — Agent Guide

## Entry Point

```python
from src.docling_agent import process_documents
result = process_documents(agent_settings: dict, *, enable_tracing: bool = False) -> dict
```

Config is a plain dict throughout — no `PDFProcessorState` TypedDict or compiled `StateGraph`.

## Workflow

1. **Extract config** (`src/graph/prompt_extraction.py`): If `user_prompt` is set, an LLM extracts overrides via `extract_config_from_prompt`; non-`None` values are applied via `merge_overrides`. Unextractable keys keep `.env` defaults. Keys in `src/graph/prompts/extractor.py`'s `_SENSITIVE_KEYS` (`folder_path`, `base_url`, `api_key`, `embedding_base_url`, `embedding_api_key`, `vlm_api_url`, `vlm_api_headers`) — plus `vlm_api_params` and vector DB secrets — are never extracted from free-text `user_prompt`: they're excluded from the extraction system prompt, and `extract_config_from_prompt`/`merge_overrides` force them to stay unset even if a model response or caller-supplied overrides dict includes them. This boundary exists because `user_prompt` is treated as untrusted input — allowing it to set network endpoints, credentials, or filesystem paths would let a crafted prompt redirect embedding/VLM traffic or read arbitrary directories.
2. **Load documents** (`src/graph/document_paths.py`): `list_document_paths(folder_path)` globs for `*.pdf`, `*.docx`, `*.xlsx`, `*.md`, `*.html`, `*.latex`, `*.tex` (sorted); `shard_paths(paths, shard_index, shard_count)` keeps `paths[shard_index::shard_count]` so parallel instances (e.g. Kubernetes Indexed Job pods) split the work without a coordinator. Both run via `_discover_paths` in `src/docling_agent.py`.
3. **Batched pipeline** (`src/graph/document_pipeline.py`): `for start in range(0, len(input_data), max_conc)` — each batch processes up to `max_conc` documents through these stages in parallel:

   | Stage | Function | Notes |
   |-------|----------|-------|
   | Convert | `convert_doc` | Fresh `DocumentConverter` per task via `_new_converter`. Standard handles all formats; VLM (`_effective_pipeline fallback to standard) applies to PDF only. Output record: `content` (markdown), `docling_document`, `metadata`. Errors → `doc: None` + `error` string. |
   | Chunk | `chunk_doc` | `fixed` → chunks markdown `content` by `chunk_size`/`chunk_overlap`. `hybrid` → uses `docling_document` + `HybridChunker.contextualize`. Metadata extended with `chunk_index`, `chunk_count`. |
   | Embed | `embed_doc` | One embedding per chunk via `embed_texts`. Transient API errors (rate limits, timeouts, connection issues, 5xx) are retried with backoff; a batch that still fails afterward → `None` at those indices plus a non-blocking `embedding_error` note on the document record. |
   | Store | `store_doc` | Writes precomputed vectors to `chroma`/`milvus` via cached connections. Supports write modes: `insert` (default), `upsert` (delete+insert), `skip_existing`. Features: parallel batch processing, retry with exponential backoff, circuit breaker for failures, connection pooling. Skips `None` embeddings. IDs: `{file_path}#chunk_{chunk_index}`. No-op when backend is `none`. |

   With `config["configurable"]["mlflow_run_id"]` set, `document_workflow` logs cumulative `progress_metrics` after every batch (`step=docs_completed`, via `MlflowClient`).
4. **Aggregate** (`src/docling_agent.py`): Flatten chunks → `documents`, vectors → `embeddings`, sum `vectors_stored`/`vectors_upserted`/`retry_count`, collect `failed_chunks`, and collect failures into `error_reports` (`{"path", "error", "stage"}` where `stage` is `pipeline`, `embedding`, or `vector_store`). `embedding` is non-blocking -- it means some (not necessarily all) of a document's chunks failed to embed after retries; `chunks`/`embeddings` still include whatever succeeded, unlike `pipeline`/`vector_store` errors which mean nothing was stored for that document. Result dict also includes all resolved `agent_settings` keys except `documents`/`embeddings`.

## Return Shape

| Key | Type | Description |
|-----|------|-------------|
| `documents` | `list[dict]` | Chunk records (content + metadata) |
| `embeddings` | `list[list[float] | None]` | One per chunk |
| `vectors_stored` | `int` | Count of vectors written |
| `vectors_upserted` | `int` | Count of vectors replaced (upsert mode) |
| `retry_count` | `int` | Total retry attempts across all batches |
| `failed_chunks` | `list[dict]` | `{"id", "error"}` for chunks that failed to store |
| `error_reports` | `list[dict]` | `{"path", "error", "stage"}` per file-level failure; `stage` is `pipeline`, `embedding`, or `vector_store` |
| remaining keys | — | From resolved `agent_settings` |

## Key Conventions

- **Extend with `@task`**: Add or modify `@task` functions in `src/graph/document_pipeline.py`, not by wiring LangGraph `StateGraph` nodes.
- **Per-file isolation**: Conversion/store failures do not abort other documents.
- **Batching**: `max_concurrency` limits parallel docs **per batch**, not across the full file list. Batch size comes from `config["configurable"]["max_concurrency"]` (validated by `_max_concurrency_from_config`; default CPU count, min 1).
- **Pipeline input**: Prefer `{"file_path": str, "config": dict}`; `pdf_path` is legacy but still accepted by `document_pipeline`.
- **Hybrid chunking**: Requires `docling_document` on the conversion record; fixed uses markdown `content`.
- **VLM**: PDF only. `CONVERSION_PIPELINE=vlm` with non-PDF files uses standard conversion via `_effective_pipeline`.

## Module Map

| Module | Role |
|--------|------|
| `src/docling_agent.py` | `process_documents`, CLI, MLflow orchestration |
| `src/graph/document_pipeline.py` | `@task`/`@entrypoint` batched pipeline: convert → chunk → embed → store |
| `src/graph/vector_store.py` | Full CRUD: `get_or_create_vector_store`, `store_chunk_embeddings` (insert/upsert), `query_vector_store`, `delete_chunk_embeddings`. Features: connection pooling, parallel batching, retry, circuit breaker |
| `src/graph/vector_operations.py` | Higher-level operations: `delete_document`, `replace_document_chunks`, `delete_documents_batch` |
| `src/graph/chunking.py` | `chunk_document_records` (fixed), `chunk_document_records_hybrid` |
| `src/graph/embeddings.py` | `embed_texts`, `embed_single_text`, singleton `get_embedding_client` with connection pooling |
| `src/graph/prompt_extraction.py` | `extract_config_from_prompt`, `merge_overrides` |
| `src/graph/prompts/extractor.py` | System prompt; NLP-extractable keys in `_CONFIG_KEYS` |
| `src/graph/document_paths.py` | `DOCUMENT_GLOBS`, `list_document_paths`, `shard_paths` |
| `src/setup/settings_spec.py` | `SETTINGS_SPEC` — single source of truth for every setting (env var, type, default, group, `choices`, `sensitive`, `untrusted`, `env_fallbacks`, `depends_on`); `SENSITIVE_SETTING_KEYS`; `settings_contract()` — the JSON contract for launchers. No third-party imports. |
| `src/setup/agent_settings.py` | `load_environment_variables(envfile=None)` — optional `.env` loading, then env→config via `SETTINGS_SPEC` |
| `src/utils/json_parse.py` | `_parse_json_object_str` for VLM headers/params env vars |
| `src/utils/uuids.py` | `generate_mlflow_run_name` (`docling_pdf_processing_{timestamp}_{uuid}`) |
| `src/utils/mlflow_tracing.py` | `session_context`, `run_context` (attaches a run to a worker thread **without ending it**), per-document span helpers |
| `src/utils/logging_setup.py` | `configure_logging(level, fmt, static_fields)` — text or JSON-lines logs (`LOG_FORMAT=json`) |
| `src/utils/circuit_breaker.py` | `CircuitBreaker` — resilience pattern for preventing cascading failures |

## MLflow Tracing

Gated by `ENABLE_TRACING` (env) / `enable_tracing` (config).

- **CLI path** (`src/docling_agent.py` `__main__`): When `enable_tracing` is truthy → `mlflow.set_tracking_uri`, `mlflow.set_experiment`, `mlflow.openai.autolog()` (`mlflow.langchain.autolog()` only with `MLFLOW_LANGCHAIN_AUTOLOG=1`: it records whole DoclingDocuments in LangGraph task spans, and their uint64 `binary_hash` makes an MLflow server reject the entire trace), unique run name (`_shard{i}` suffix when sharded), session grouping via `session_id = mlflow_session_id or mlflow_parent_run_id or run.info.run_id`. With `mlflow_parent_run_id` set, the run is a nested child of that run (one shard of a scheduler-launched logical run). Tags: `shard_index`, `shard_count`, `total_document_count`, `scheduler_run_id`/`scheduler_executor` when set. Spans: `NLP prompt extraction`, `document discovery`, `Parallel pipeline`, `Results aggregation`. Metrics: `document_count` (this shard), plus per-batch `progress_metrics` with `step=docs_completed` (`docs_completed`, `docs_failed`, `total_chunks`, `total_embeddings`, `non_null_embeddings`, `vectors_stored`, `vectors_upserted`, `vector_store_retry_count`, `failed_chunk_count`, `error_count`, `vector_store_error_count`) — the latest value is the final total. Artifacts: `run_summary.json` always, `error_reports.json` when non-empty. `SENSITIVE_SETTING_KEYS` (plus a credential-name regex) are never logged as params.
- **Per-document traces**: root `document_pipeline` span per document with attributes `file`, `file_format`, `shard_index`, `chunk_count`, `stored_count`, `embedding_failed`, and `stage_failed` on errors.
- **Run status**: `run_context` must never end the run (MLflow's `with start_run(run_id=...)` would mark it FINISHED mid-pipeline); the native run status is the source of truth for schedulers.
- **Task spans**: Workers must have `ENABLE_TRACING=1` in env at **import time** (before loading `graph`/`pipeline` modules). Spans: `convert_pdf`, `chunk_doc`, `embed_doc`, `store_doc` + helpers `chunk_fixed`/`chunk_hybrid`/`embed_texts`/`list_documents`.
- **Session filter**: ``metadata.`mlflow.trace.session` = '<run_id or MLFLOW_SESSION_ID>'``.
- `process_documents(..., enable_tracing=True)` starts a top-level MLflow run. The config dict key alone does not enable tracing unless the flag is passed.

## CLI and container

- `python -m src.docling_agent [--envfile PATH] [--user_prompt TEXT] [--print-settings-schema]` — `--envfile` defaults to `./.env` when present; otherwise configuration comes purely from the environment. `--user_prompt` overrides `USER_PROMPT`.
- Exit codes: `0` completed (even with per-file errors in `error_reports`), `1` fatal error, `2` configuration error (missing env file, invalid JSON setting, shard index out of range).
- `Containerfile`: multi-stage `uv` build with `--extra vector-stores`, docling models pre-downloaded (`DOCLING_ARTIFACTS_PATH`), non-root, arbitrary-UID friendly; defaults `FOLDER_PATH=/data/in`, `LOG_FORMAT=json`.

## Settings contract and launchers

This repo hosts only the agent. The scheduler ([docling-scheduler](https://github.com/mcaimi/docling-scheduler), Podman only) and its web UI ([docling-ui](https://github.com/mcaimi/docling-ui)) are separate projects that **never import agent code**. They depend only on the image, the environment variables, MLflow (run hierarchy, tags, `step=docs_completed` metrics, per-document traces, artifacts) and the exit codes described above.

- `settings_contract(document_globs)` in `src/setup/settings_spec.py` serialises `SETTINGS_SPEC` plus `DOCUMENT_GLOBS` (`src/graph/document_paths.py`) as JSON: `schema_version` (`CONTRACT_VERSION`, bump only on incompatible shape changes), `document_globs`, `settings`. Sensitive defaults are `null`; computed defaults are named in `dynamic_default` (`max_concurrency` → `cpu_count`) instead of being resolved on the generating machine.
- `python -m src.docling_agent --print-settings-schema` prints it; `settings-schema.json` at the repo root is the committed copy, kept current by `tests/unit/test_settings_contract.py`. After changing a setting or the globs, regenerate it and copy it to `docling-scheduler/src/docling_scheduler/settings-schema.json`.

## Env → Config Mapping (key vars)

| Env Var | Config Key | Default |
|---------|------------|---------|
| `FOLDER_PATH` | `folder_path` | ./pdfs |
| `CONVERSION_PIPELINE` | `conversion_pipeline` | standard |
| `CHUNKING_STRATEGY` | `chunking_strategy` | fixed |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `chunk_size` / `chunk_overlap` | 1000 / 100 |
| `MAX_CONCURRENCY` | `max_concurrency` | CPU count |
| `USER_PROMPT` | `user_prompt` | (empty) — untrusted |
| `SHARD_INDEX` (fallback `JOB_COMPLETION_INDEX`) / `SHARD_COUNT` | `shard_index` / `shard_count` | 0 / 1 |
| `LOG_LEVEL` / `LOG_FORMAT` | `log_level` / `log_format` | INFO / text |
| **Vector Store** | | |
| `VECTOR_STORE_BACKEND` | `vector_store_backend` | none |
| `VECTOR_STORE_COLLECTION` | `vector_store_collection` | docling_chunks |
| `VECTOR_STORE_WRITE_MODE` | `vector_store_write_mode` | insert |
| `VECTOR_STORE_BATCH_SIZE` | `vector_store_batch_size` | 200 |
| `VECTOR_STORE_MAX_BATCH_WORKERS` | `vector_store_max_batch_workers` | 4 |
| `VECTOR_STORE_RETRY_ATTEMPTS` | `vector_store_retry_attempts` | 3 |
| `VECTOR_STORE_CIRCUIT_BREAKER_THRESHOLD` | `vector_store_circuit_breaker_threshold` | 5 |
| **MLflow** | | |
| `ENABLE_TRACING` | `enable_tracing` | 0 |
| `MLFLOW_TRACKING_URI` | `mlflow_tracking_uri` | http://localhost:5000 |
| `MLFLOW_EXPERIMENT_NAME` | `mlflow_experiment_name` | Docling Agent |
| `MLFLOW_SESSION_ID` | `mlflow_session_id` | (unset) |
| `MLFLOW_PARENT_RUN_ID` | `mlflow_parent_run_id` | (unset) |
| `MLFLOW_LANGCHAIN_AUTOLOG` | `mlflow_langchain_autolog` | 0 (opt-in; can drop traces on an MLflow server) |
| `SCHEDULER_RUN_ID` / `SCHEDULER_EXECUTOR` | `scheduler_run_id` / `scheduler_executor` | (unset) — set by the scheduler |

The full list lives in `src/setup/settings_spec.py` (add new settings there — the loader, MLflow redaction and the settings contract all derive from it; then regenerate `settings-schema.json`). Chroma/Milvus connection settings (`CHROMA_*`, `MILVUS_*`) are env-only.

## Vector Store Features (New)

### CRUD Operations
- **Create/Insert**: `store_chunk_embeddings(..., write_mode="insert")` — default behavior
- **Update/Upsert**: `store_chunk_embeddings(..., write_mode="upsert")` — deletes existing IDs, then inserts (idempotent)
- **Read/Query**: `query_vector_store(vector_store, query, k=5, ...)` — similarity search with optional metadata filters
- **Delete**: `delete_chunk_embeddings(vector_store, ids=..., file_paths=..., metadata_filters=...)`

### Performance & Resilience
- **Connection pooling**: Vector store instances cached and reused (similar to embedding client)
- **Parallel batch processing**: Configurable worker threads (`VECTOR_STORE_MAX_BATCH_WORKERS`)
- **Retry logic**: Exponential backoff for transient failures (connection errors, timeouts)
- **Circuit breaker**: Prevents cascading failures after threshold consecutive errors
- **Enhanced metrics**: MLflow tracks throughput (vectors/sec), retry counts, batch durations, upsert counts

### Write Modes
- `insert`: Create new records (default). Duplicate ID behavior is backend-dependent
- `upsert`: Delete existing records with matching IDs, then insert (guarantees idempotency)
- `skip_existing`: Only insert chunks whose IDs don't already exist

### Backend-Specific Enhancements
**Milvus**: Supports `MILVUS_INDEX_TYPE` (HNSW, IVF_FLAT), `MILVUS_METRIC_TYPE` (COSINE, L2, IP), `MILVUS_CONSISTENCY_LEVEL` (Strong, Bounded, Eventually), `MILVUS_PARTITION_NAME`

**ChromaDB**: Supports `CHROMA_DISTANCE_FUNCTION` (cosine, l2, ip), `CHROMA_COLLECTION_METADATA`
