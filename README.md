# PDF Processing Agent

A multi-step workflow built with **LangGraph** that processes PDF files using **docling** for conversion and an **OpenAI-compatible** HTTP API for embeddings.

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) or another PEP 517 installer

## Features

- Load PDF files from a specified folder
- Convert PDFs to text using docling
- Chunk documents into smaller parts (size and overlap are configured in `src/config.py`)
- Generate embeddings for each chunk via an OpenAI-compatible embeddings endpoint

## Installation

```bash
uv sync
```

## Setup

1. Configure environment variables (for example in a `.env` file in the project root). The embedding client reads:

   - `EMBEDDING_API_KEY`: API key for the embedding endpoint (or `OPENAI_API_KEY` as a fallback)
   - `EMBEDDING_BASE_URL`: Base URL of the OpenAI-compatible embedding API (optional; omit for the default OpenAI host)
   - `PDF_FOLDER_PATH`: Directory containing PDF files to process (optional; defaults to `./pdfs` in `src/config.py`)

   If you use a `.env` file, load it from your own script (for example `from dotenv import load_dotenv; load_dotenv()`), or export the variables in your shell before running.

2. Put PDFs in that folder (or point `PDF_FOLDER_PATH` at an existing folder).

## Usage

```python
from src.pdf_agent import process_pdfs

result = process_pdfs("./path/to/pdf/folder")
# result is a dict (TypedDict) with keys:
# folder_path, pdf_files, documents, embeddings
chunks = result["documents"]
vectors = result["embeddings"]
```

Run the bundled entrypoint (uses `PDF_FOLDER_PATH` from the environment or the default folder):

```bash
uv run python -m src.pdf_agent
```

## Workflow

The LangGraph pipeline runs these steps in order:

1. **Load PDFs**: Discover `*.pdf` files in the folder
2. **Convert PDFs**: Turn each file into text (and basic metadata) with docling
3. **Chunk documents**: Split text into overlapping chunks
4. **Generate embeddings**: Call the embeddings API for each chunk (requires a valid API key when this step runs)

## Dependencies

Declared in `pyproject.toml`:

- `langgraph` (1.2+)
- `docling` (2.93+)
- `openai` (1.14+)
- `python-dotenv` (1.2+)

## Architecture

### Workflow Graph

```mermaid
flowchart LR
    START((START)) --> load["load_pdfs<br/>(list PDF files)"]
    load --> convert["convert_pdfs<br/>(docling conversion)"]
    convert --> chunk["chunk_documents<br/>(split + overlap)"]
    chunk --> embed["generate_embeddings<br/>(embedding API call)"]
    embed --> END((END))
```

### Class Diagram

```mermaid
classDiagram
    class PDFProcessorState {
        <<TypedDict>>
        +str folder_path
        +str api_key
        +str embedding_model
        +dict connection_params
        +NotRequired[int] chunk_size
        +NotRequired[int] chunk_overlap
        +list[str] pdf_files
        +list[dict] documents
        +list[float | None] embeddings
        +NotRequired[str] conversion_pipeline
        +NotRequired[str] vlm_api_url
        +NotRequired[dict] vlm_api_headers
        +NotRequired[dict] vlm_api_params
        +NotRequired[str] vlm_preset
    }

    class pdf_agent {
        +create_pdf_processor_graph() CompiledGraph
        +process_pdfs(folder_path, conversion_pipeline) PDFProcessorState
    }

    class graph_nodes {
        +load_pdfs(PDFProcessorState) dict
        +convert_pdfs(PDFProcessorState) dict
        +chunk_documents(PDFProcessorState) dict
        +generate_embeddings(PDFProcessorState) dict
    }

    class agent_settings {
        +str EMBEDDING_MODEL
        +int EMBEDDING_DIMENSIONS
        +int CHUNK_SIZE
        +int CHUNK_OVERLAP
        +str PDF_FOLDER_PATH
        +str EMBEDDING_BASE_URL
        +str EMBEDDING_API_KEY
        +str DOCLING_PDF_PIPELINE_ENV
        +str DOCLING_VLM_API_URL_ENV
        ...
    }

    class config {
        +str EMBEDDING_MODEL "text-embedding-ada-002"
        +int EMBEDDING_DIMENSIONS 1536
        +int CHUNK_SIZE 1000
        +int CHUNK_OVERLAP 100
        +str PDF_FOLDER_PATH "./pdfs"
        ...
    }

    class _parse_json_object_str {
        +str json_str ~ dict | None
    }

    pdf_agent --> graph_nodes : adds nodes to graph
    pdf_agent --> PDFProcessorState : operates on
    graph_nodes --> PDFProcessorState : updates
    pdf_agent --> agent_settings : reads settings
    agent_settings ..|> config : overrides defaults
    pdf_agent --> _parse_json_object_str : uses helper
    graph_nodes <-- pdf_conversion : convert_pdf_paths_to_documents
    graph_nodes <-- pdf_paths : list_pdf_paths
    graph_nodes <-- chunking : chunk_document_records
    graph_nodes <-- embeddings : embed_texts
```

<!-- existing dependency section for completeness -->
## Dependencies

Declared in `pyproject.toml`:

- `langgraph` (1.2+)
- `docling` (2.93+)
- `openai` (1.14+)
- `python-dotenv` (1.2+)

## License

GPL-3

## AI Disclosure

This project has been developed with the help of an AI Model. Specifically an instance of Qwen3.6-35B-A3B running in vLLM
