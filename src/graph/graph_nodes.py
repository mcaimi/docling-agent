#!/usr/bin/env python

"""LangGraph node callables for the PDF processing workflow."""

from typing import Any

# import required modules
# and especially load the agent settings module
try:
    from .chunking import chunk_document_records
    from .embeddings import embed_texts
    from .graph_state import PDFProcessorState
    from .pdf_conversion import convert_pdf_paths_to_documents
    from .pdf_paths import list_pdf_paths
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# AGENT NODE CALLABLES

# load the PDF files from the folder
def load_pdfs(state: PDFProcessorState) -> dict[str, Any]:
    paths = list_pdf_paths(state["folder_path"])
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
    vectors = embed_texts(texts, state["embedding_model"], state["connection_params"])
    print(f"Generated embeddings for {len(vectors)} chunks")
    return {"embeddings": vectors}

# export the node callables
__all__ = ["load_pdfs", "convert_pdfs", "chunk_documents", "generate_embeddings"]