# Docling Document Processing Agent

LangGraph workflow that discovers documents (PDF, DOCX, XLSX, Markdown, HTML, LaTeX), converts them in **parallel batches** with **docling**, chunks text with overlap, requests **OpenAI-compatible** embeddings per chunk, and optionally persists vectors to **Chroma** or **Milvus** via LangChain connectors. Optional **MLflow** tracing records runs, spans, and metrics when enabled.

## Setup

1. **Python 3.11+** and [uv](https://docs.astral.sh/uv/) (recommended).

2. Install dependencies:

   ```bash
   uv sync
   # Optional vector DB backends:
   uv sync --extra chroma
   uv sync --extra milvus
   uv sync --extra vector-stores
   ```

   Install the matching extra before setting `VECTOR_STORE_BACKEND` to `chroma` or `milvus`.

3. **Environment**: Create a `.env` file in the project root (see `.env` for the full list of configurable variables). All settings can also be overridden via shell environment variables.

4. **MLflow (optional)**: When `ENABLE_TRACING=1`, start an MLflow tracking server (default `MLFLOW_TRACKING_URI=http://localhost:5000`) and open the experiment named in `MLFLOW_EXPERMENT_NAME` (default **Docling Agent**) to inspect runs. Pipeline worker modules read `ENABLE_TRACING` from the environment at import time, so load `.env` (or export the variable) before importing graph modules if per-task spans should include `store_doc`.

## Usage

```python
from src.setup import load_environment_variables
from src.docling_agent import process_documents

agent_settings = load_environment_variables(".env")
# or pass a config dict directly:
# agent_settings = {
#     "folder_path": "./pdfs",
#     "user_prompt": "Use VLM pipeline and hybrid chunking with 500 char chunks",  # optional NLP extraction
#     "embedding_model": "text-embedding-ada-002",
#     "embedding_dimensions": 1536,
#     "embedding_base_url": "https://api.openai.com/v1",
#     "embedding_api_key": "your-api-key-here",
#     "conversion_pipeline": "standard",  # or "vlm" (PDF only)
#     "chunking_strategy": "fixed",        # or "hybrid"
#     "chunk_size": 1000,
#     "chunk_overlap": 100,
#     "max_concurrency": 4,
#     "vector_store_backend": "none",      # or "chroma", "milvus" (requires matching uv extra)
#     "vector_store_collection": "docling_chunks",
#     # Chroma/Milvus connection settings (CHROMA_*, MILVUS_*) are env-only, not NLP-extracted
#     "enable_tracing": 0,
#     "mlflow_tracking_uri": "http://localhost:5000",
#     "mlflow_experiment_name": "Docling Agent",
# }

result = process_documents(agent_settings)
# result merges resolved agent_settings with:
#   documents      — list of chunk records (content + metadata)
#   embeddings     — list[list[float] | None], one per chunk
#   vectors_stored — count of chunk vectors written when vector_store_backend is chroma/milvus
#   error_reports  — list of {"path", "error", "stage"} for pipeline or vector_store failures
chunks = result["documents"]
vectors = result["embeddings"]
backend = (result.get("vector_store_backend") or "none").lower()
if backend != "none":
    print(f"Vectors stored in {backend}: {result.get('vectors_stored', 0)}")
if result.get("error_reports"):
    for err in result["error_reports"]:
        print(f"Failed ({err.get('stage', 'pipeline')}): {err['path']}: {err['error']}")
```

CLI (uses `FOLDER_PATH` and other variables from the `.env` file or the configured defaults):

```bash
uv run python -m src.docling_agent
# Custom env file:
uv run python -m src.docling_agent --envfile .env.local
# User prompt for NLP extraction:
uv run python -m src.docling_agent --user_prompt "Use VLM pipeline and hybrid chunking"
```

With `ENABLE_TRACING=1`, the CLI connects to `mlflow_tracking_uri`, uses `mlflow_experiment_name`, and starts a uniquely named run (`docling_pdf_processing_{timestamp}_{uuid}` via `generate_mlflow_run_name`). All traces from that invocation are grouped under one MLflow session (default: the run’s `run_id`; override with `MLFLOW_SESSION_ID`). It logs sanitized config params, workflow spans (NLP extraction, document discovery, pipeline, aggregation), metrics (`document_count`, `total_chunks`, `total_embeddings`, `vectors_stored`, `error_count`, `vector_store_error_count`, `non_null_embeddings`), and writes `error_reports.json` when any file fails. When a vector store backend is configured, the CLI also prints `Vectors stored in {backend}: N`. Per-document tasks also emit spans when `ENABLE_TRACING` is set in the environment before worker modules load.

## Environment Variables

| Variable | Default | Role |
|---|---|---|
| `FOLDER_PATH` | `./pdfs` | Directory containing documents to process |
| `EMBEDDING_API_KEY` | `your-api-key-here` | API key for the embedding endpoint |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Base URL of the OpenAI-compatible API |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Model id passed to the embeddings API |
| `EMBEDDING_DIMENSIONS` | `1536` | Expected vector size for the chosen model |
| `CHUNK_SIZE` | `1000` | Characters per fixed-size chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive fixed-size chunks |
| `CONVERSION_PIPELINE` | `standard` | `standard` (multi-format docling pipeline) or `vlm` (PDF only) |
| `CHUNKING_STRATEGY` | `fixed` | `fixed` (character-based) or `hybrid` (document-aware) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model for NLP extraction |
| `BASE_URL` | `https://api.openai.com/v1` | Base URL for the LLM |
| `API_KEY` | `your-api-key-here` | API key for the LLM |
| `VLM_API_URL` | `https://api.openai.com/v1` | Remote VLM endpoint URL (required when `CONVERSION_PIPELINE=vlm`) |
| `VLM_API_HEADERS` | `{}` | JSON object of headers for the VLM API |
| `VLM_API_PARAMS` | `{}` | JSON object of params for the VLM API |
| `VLM_PRESET` | `smoldocling` | Preset name for the VLM conversion |
| `MAX_CONCURRENCY` | CPU count (`os.cpu_count()`) | Max documents processed in parallel per batch |
| `ENABLE_TRACING` | `0` | Set to `1` to enable MLflow tracing in the CLI and pipeline modules |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL (CLI tracing path) |
| `MLFLOW_EXPERIMENT_NAME` | `Docling Agent` | MLflow experiment name |
| `MLFLOW_SESSION_ID` | _(unset)_ | Optional trace session id; CLI default is the active run’s `run_id` |
| `VECTOR_STORE_BACKEND` | `none` | `none`, `chroma`, or `milvus` (opt-in persistence) |
| `VECTOR_STORE_COLLECTION` | `docling_chunks` | Collection name in Chroma/Milvus |
| `CHROMA_HOST` | _(unset)_ | Remote Chroma server host (optional) |
| `CHROMA_PORT` | `8000` | Remote Chroma server port |
| `CHROMA_PERSIST_DIRECTORY` | _(unset)_ | Local Chroma persistence directory |
| `MILVUS_URI` | `http://localhost:19530` | Milvus server URI or Milvus Lite file path |
| `MILVUS_TOKEN` | _(unset)_ | Milvus auth token |
| `MILVUS_DB_NAME` | `default` | Milvus database name |
| `VECTOR_STORE_BATCH_SIZE` | `200` | Max vectors per insert sub-batch |

Vector persistence uses **precomputed** embeddings from the pipeline (no second embed call). Chunk IDs are stable: `{file_path}#chunk_{chunk_index}`. Chroma/Milvus connection settings are env-only and are not extracted from natural-language prompts.

## Supported formats

| Pipeline | Formats |
|---|---|
| `standard` | PDF, DOCX, XLSX, Markdown (`.md`), HTML, LaTeX (`.latex`, `.tex`) |
| `vlm` | PDF only (non-PDF files fall back to the standard pipeline) |

Discovery globs in `list_document_paths` match the standard pipeline set above.

## Architecture

Agent and contributor details: see [AGENTS.md](AGENTS.md).

```mermaid
flowchart TB
    subgraph orchestration [process_documents / CLI]
        extract[NLP config extraction]
        listDocs[List document paths]
        invoke["document_workflow<br/>batched by max_concurrency"]
        extract --> listDocs --> invoke
    end
    subgraph perDoc [parallel per document in batch]
        convert[Convert]
        chunk[Chunk]
        embed[Embed]
        store[Store]
        convert --> chunk --> embed --> store
    end
    invoke --> perDoc
    perDoc --> merge["Merge chunks, embeddings,<br/>vectors_stored, error_reports, config"]
    merge -.-> mlflow[MLflow run optional]
```

## Workflow

1. **Extract config** (optional): If a `user_prompt` is provided (via CLI `--user_prompt` or the `user_prompt` config field), an LLM extracts configuration overrides. Unextractable parameters keep their `.env` defaults.
2. **Load documents**: Discover supported files in `FOLDER_PATH` via `list_document_paths`.
3. **Process in batches**: Documents are handled in groups of up to `MAX_CONCURRENCY`. Within each batch, convert → chunk → embed → store run in parallel for every file in that batch; the next batch starts when the current one finishes. The store step is skipped when `vector_store_backend` is `none`. Conversion supports `standard` (multi-format) or `vlm` (PDF only); chunking supports `fixed` (character-based with `CHUNK_SIZE`/`CHUNK_OVERLAP`) or `hybrid` (docling `HybridChunker`).
4. **Aggregate results**: All chunks and embeddings are merged into `documents` and `embeddings`, with `vectors_stored` summing successful writes. Per-file failures are collected in `error_reports` with `stage` of `pipeline` or `vector_store`, without stopping other files. The returned dict also includes the resolved agent settings (e.g. `folder_path`, `chunking_strategy`) except `documents` and `embeddings`.

## MLflow tracing

| When | What is recorded |
|---|---|
| CLI with `enable_tracing` / `ENABLE_TRACING=1` | Top-level run, LangChain/OpenAI autolog, workflow spans, metrics, `error_reports.json` artifact |
| Pipeline modules (`ENABLE_TRACING=1` in env) | Spans on convert, chunk, embed, store (`store_doc`), document listing, and chunking helpers |

**Session grouping:** The CLI sets `mlflow_session_id` to `MLFLOW_SESSION_ID` or the active run’s `run_id`, wraps the instrumented workflow in `mlflow.tracing.context(session_id=...)`, and passes the id in per-document config so parallel tasks tag traces the same way. Filter in the MLflow UI: ``metadata.`mlflow.trace.session` = '<session_id>'``.

Sensitive values are not logged as MLflow params: `api_key`, `embedding_api_key`, `vlm_api_headers`, `vlm_api_params`.

Configure the tracking server and experiment via `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERMENT_NAME` in `.env` (mapped to `mlflow_tracking_uri` and `mlflow_experiment_name` in the config dict). Pass `enable_tracing=True` to `process_documents()` to start a top-level MLflow run; the config dict key alone does not enable tracing unless you pass the flag (the CLI maps `enable_tracing` from settings).

## Module structure

- `src/docling_agent.py` — entry point (`process_documents`, CLI, optional MLflow orchestration)
- `src/graph/document_pipeline.py` — batched parallel per-document convert/chunk/embed/store pipeline
- `src/graph/vector_store.py` — `get_vector_store`, `store_chunk_embeddings` (Chroma/Milvus, precomputed vectors)
- `src/graph/chunking.py` — fixed-size and hybrid chunking
- `src/graph/embeddings.py` — OpenAI-compatible embedding client
- `src/graph/prompt_extraction.py` — NLP config extraction from natural language
- `src/graph/document_paths.py` — document discovery
- `src/setup/agent_settings.py` — `.env` loading
- `src/utils/uuids.py` — `generate_mlflow_run_name` for unique CLI run names
- `src/utils/mlflow_tracing.py` — `session_context` for MLflow trace session grouping

## Dependencies

Declared in `pyproject.toml`:

- `docling>=2.93.0`
- `langchain>=1.3.1` (MLflow LangChain autolog)
- `langgraph>=1.2.0`
- `mlflow>=3.12.0`
- `openai>=2.30.0`
- `python-dotenv>=1.2.2`

Optional extras:

- `chroma`: `langchain-chroma`
- `milvus`: `langchain-milvus`
- `vector-stores`: both backends

## License

GPL-3

## AI Disclosure

This project has been developed with the help of an AI Model. Specifically an instance of Qwen3.6-35B-A3B running in vLLM
