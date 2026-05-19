#!/usr/bin/env python

"""LangGraph node callables for the PDF processing workflow."""

from typing import Any

# import required modules
# and especially load the agent settings module
try:
    from .chunking import chunk_document_records, chunk_document_records_hybrid
    from .embeddings import embed_texts
    from .graph_state import PDFProcessorState
    from .pdf_conversion import convert_pdf_paths_to_documents
    from .pdf_paths import list_pdf_paths
    from .prompt_extraction import extract_config_from_prompt as _extract_config_llm
    from .prompt_extraction import merge_overrides
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# AGENT NODE CALLABLES


# load the PDF files from the folder
def load_pdfs(state: PDFProcessorState) -> dict[str, Any]:
    paths = list_pdf_paths(state.get("folder_path", "./pdfs"))
    print(f"Found {len(paths)} PDF files")
    return {"pdf_files": paths}


# convert the PDF files to DoclingDocuments
def convert_pdfs(state: PDFProcessorState) -> dict[str, Any]:
    documents = convert_pdf_paths_to_documents(
        state.get("pdf_files"),
        pipeline=state.get("conversion_pipeline", "standard"),
        vlm_api_url=state.get("vlm_api_url", None),
        vlm_api_headers=state.get("vlm_api_headers", None),
        vlm_api_params=state.get("vlm_api_params", None),
        vlm_preset=state.get("vlm_preset", None),
    )
    print(f"Converted {len(documents)} documents")
    return {"documents": documents}


# chunk the documents (Docling Chunker and Hybrid Chunker)
def chunk_documents(state: PDFProcessorState) -> dict[str, Any]:
    strategy = state.get("chunking_strategy", "fixed")
    print(f"Chunking strategy: {strategy}")
    if strategy == "hybrid":
        chunked = chunk_document_records_hybrid(state["documents"])
    else:  # "fixed"
        chunked = chunk_document_records(
            state["documents"],
            state["chunk_size"],
            state["chunk_overlap"],
        )
    print(f"Chunked documents into {len(chunked)} chunks")
    return {"documents": chunked}


# use a remote OpenAI-compatible embedding API to generate embeddings for the chunks
def generate_embeddings(state: PDFProcessorState) -> dict[str, Any]:
    texts = [doc["content"] for doc in state["documents"]]
    vectors = embed_texts(
        texts,
        state["embedding_model"],
        state["embedding_base_url"],
        state["embedding_api_key"],
    )
    print(f"Generated embeddings for {len(vectors)} chunks")
    return {"embeddings": vectors}


# extract configuration parameters from a natural language user prompt using an LLM
def extract_config_from_prompt(state: PDFProcessorState) -> dict[str, Any]:
    user_prompt = state.get("user_prompt", "")
    model = state.get("llm_model", "gpt-4o-mini")
    base_url = state.get("base_url", "https://api.openai.com/v1")
    api_key = state.get("api_key", "")
    config = _extract_config_llm(user_prompt, model, base_url, api_key)
    # Filter out nulls so only extracted values override .env defaults in state
    filtered = {k: v for k, v in config.items() if v is not None}

    # apply overrides
    return merge_overrides(state, filtered)


# export the node callables
__all__ = [
    "load_pdfs",
    "convert_pdfs",
    "chunk_documents",
    "generate_embeddings",
    "extract_config_from_prompt",
]
