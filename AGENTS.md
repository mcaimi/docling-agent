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
   | Store | `store_doc` | When `vector_store_backend` is `chroma`/`milvus`, writes precomputed vectors via LangChain connectors. Skips `None` embeddings. IDs: `{file_path}#chunk_{chunk_index}`. No-op when backend is `none`. |

4. **Aggregate** (`src/docling_agent.py`): Flatten chunks → `documents`, vectors → `embeddings`, sum `vectors_stored`, collect failures into `error_reports` (`{"path", "error", "stage"}` where `stage` is `pipeline` or `vector_store`). Result dict also includes all resolved `agent_settings` keys except `documents`/`embeddings`.

## Return Shape

| Key | Type | Description |
|-----|------|-------------|
| `documents` | `list[dict]` | Chunk records (content + metadata) |
| `embeddings` | `list[list[float] | None]` | One per chunk |
| `vectors_stored` | `int` | Count of vectors written |
| `error_reports` | `list[dict]` | `{"path", "error", "stage"}` per failure |
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
| `src/graph/vector_store.py` | `get_vector_store`, `store_chunk_embeddings` (Chroma/Milvus, precomputed vectors) |
| `src/graph/chunking.py` | `chunk_document_records` (fixed), `chunk_document_records_hybrid` |
| `src/graph/embeddings.py` | `embed_texts`, singleton `get_embedding_client` |
| `src/graph/prompt_extraction.py` | `extract_config_from_prompt`, `merge_overrides` |
| `src/graph/prompts/extractor.py` | System prompt; NLP-extractable keys in `_CONFIG_KEYS` |
| `src/graph/document_paths.py` | `list_document_paths` — supported extension globs |
| `src/setup/agent_settings.py` | `load_environment_variables` — `.env` loading, env→config mapping |
| `src/utils/json_parse.py` | `_parse_json_object_str` for VLM headers/params env vars |
| `src/utils/uuids.py` | `generate_mlflow_run_name` (`docling_pdf_processing_{timestamp}_{uuid}`) |
| `src/utils/mlflow_tracing.py` | `session_context` — wrapper around `mlflow.tracing.context(session_id=...)` |

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
| `VECTOR_STORE_BACKEND` | `vector_store_backend` | none |
| `VECTOR_STORE_COLLECTION` | `vector_store_collection` | docling_chunks |
| `ENABLE_TRACING` | `enable_tracing` | 0 |
| `MLFLOW_TRACKING_URI` | `mlflow_tracking_uri` | http://localhost:5000 |
| `MLFLOW_EXPERIMENT_NAME` | `mlflow_experiment_name` | Docling Agent |
| `MLFLOW_SESSION_ID` | `mlflow_session_id` | (unset) |

Full env list is in the `.env` file. Chroma/Milvus connection settings (`CHROMA_*`, `MILVUS_*`) are env-only.
