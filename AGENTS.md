# Docling Document Processing Agent — Agent Guide

## Entry Point

```python
from src.docling_agent import process_documents
result = process_documents(agent_settings: dict, *, enable_tracing: bool = False) -> dict
```

Config is a plain dict throughout — no `PDFProcessorState` TypedDict or compiled `StateGraph`.

## Workflow

1. **Extract config** (`src/graph/prompt_extraction.py`): If `user_prompt` is set, an LLM extracts overrides via `extract_config_from_prompt`; non-`None` values are applied via `merge_overrides`. Unextractable keys keep `.env` defaults. `vlm_api_params` and vector DB secrets are env-only and never extracted.
2. **Load documents** (`src/graph/document_paths.py`): `list_document_paths(folder_path)` globs for `*.pdf`, `*.docx`, `*.xlsx`, `*.md`, `*.html`, `*.latex`, `*.tex`.
3. **Batched pipeline** (`src/graph/document_pipeline.py`): `for start in range(0, len(input_data), max_conc)` — each batch processes up to `max_conc` documents through these stages in parallel:

   | Stage | Function | Notes |
   |-------|----------|-------|
   | Convert | `convert_doc` | Fresh `DocumentConverter` per task via `_new_converter`. Standard handles all formats; VLM (`_effective_pipeline fallback to standard) applies to PDF only. Output record: `content` (markdown), `docling_document`, `metadata`. Errors → `doc: None` + `error` string. |
   | Chunk | `chunk_doc` | `fixed` → chunks markdown `content` by `chunk_size`/`chunk_overlap`. `hybrid` → uses `docling_document` + `HybridChunker.contextualize`. Metadata extended with `chunk_index`, `chunk_count`. |
   | Embed | `embed_doc` | One embedding per chunk via `embed_texts`. API errors → `None` at that index. |
   | Store | `store_doc` | Writes precomputed vectors to `chroma`/`milvus` via cached connections. Supports write modes: `insert` (default), `upsert` (delete+insert), `skip_existing`. Features: parallel batch processing, retry with exponential backoff, circuit breaker for failures, connection pooling. Skips `None` embeddings. IDs: `{file_path}#chunk_{chunk_index}`. No-op when backend is `none`. |

4. **Aggregate** (`src/docling_agent.py`): Flatten chunks → `documents`, vectors → `embeddings`, sum `vectors_stored`, collect failures into `error_reports` (`{"path", "error", "stage"}` where `stage` is `pipeline` or `vector_store`). Result dict also includes all resolved `agent_settings` keys except `documents`/`embeddings`.

## Return Shape

| Key | Type | Description |
|-----|------|-------------|
| `documents` | `list[dict]` | Chunk records (content + metadata) |
| `embeddings` | `list[list[float] | None]` | One per chunk |
| `vectors_stored` | `int` | Count of vectors written |
| `vectors_upserted` | `int` | Count of vectors replaced (upsert mode) |
| `retry_count` | `int` | Total retry attempts across all batches |
| `failed_chunks` | `list[dict]` | `{"id", "error"}` for chunks that failed to store |
| `error_reports` | `list[dict]` | `{"path", "error", "stage"}` per file-level failure |
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
| `src/graph/document_paths.py` | `list_document_paths` — supported extension globs |
| `src/setup/agent_settings.py` | `load_environment_variables` — `.env` loading, env→config mapping (50+ config keys) |
| `src/utils/json_parse.py` | `_parse_json_object_str` for VLM headers/params env vars |
| `src/utils/uuids.py` | `generate_mlflow_run_name` (`docling_pdf_processing_{timestamp}_{uuid}`) |
| `src/utils/mlflow_tracing.py` | `session_context` — wrapper around `mlflow.tracing.context(session_id=...)` |
| `src/utils/circuit_breaker.py` | `CircuitBreaker` — resilience pattern for preventing cascading failures |

## MLflow Tracing

Gated by `ENABLE_TRACING` (env) / `enable_tracing` (config).

- **CLI path** (`src/docling_agent.py` `__main__`): When `enable_tracing` is truthy → `mlflow.set_tracking_uri`, `mlflow.set_experiment`, `mlflow.langchain.autolog()`, `mlflow.openai.autolog()`, unique run name, session grouping via `session_id = mlflow_session_id or run.info.run_id`. Spans: `NLP prompt extraction`, `document discovery`, `Parallel pipeline`, `Results aggregation`. Metrics: `document_count`, `total_chunks`, `total_embeddings`, `vectors_stored`, `error_count`, `vector_store_error_count`, `non_null_embeddings`. Artifact: `error_reports.json` when non-empty. Secrets (`api_key`, `embedding_api_key`, `vlm_api_headers`, `vlm_api_params`) are never logged as params.
- **Task spans**: Workers must have `ENABLE_TRACING=1` in env at **import time** (before loading `graph`/`pipeline` modules). Spans: `convert_pdf`, `chunk_doc`, `embed_doc`, `store_doc` + helpers `chunk_fixed`/`chunk_hybrid`/`embed_texts`/`list_documents`.
- **Session filter**: ``metadata.`mlflow.trace.session` = '<run_id or MLFLOW_SESSION_ID>'``.
- `process_documents(..., enable_tracing=True)` starts a top-level MLflow run. The config dict key alone does not enable tracing unless the flag is passed.

## Env → Config Mapping (key vars)

| Env Var | Config Key | Default |
|---------|------------|---------|
| `FOLDER_PATH` | `folder_path` | ./pdfs |
| `CONVERSION_PIPELINE` | `conversion_pipeline` | standard |
| `CHUNKING_STRATEGY` | `chunking_strategy` | fixed |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `chunk_size` / `chunk_overlap` | 1000 / 100 |
| `MAX_CONCURRENCY` | `max_concurrency` | CPU count |
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

Full env list (50+ vars) is in the `.env` file. Chroma/Milvus connection settings (`CHROMA_*`, `MILVUS_*`) are env-only.

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
