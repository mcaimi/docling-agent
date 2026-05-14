"""LangGraph node callables for the PDF processing workflow."""

from typing import Any

from src.chunking import chunk_document_records
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from src.embeddings import embed_texts
from src.graph_state import PDFProcessorState
from src.pdf_conversion import convert_pdf_paths_to_documents
from src.pdf_paths import list_pdf_paths


def load_pdfs(state: PDFProcessorState) -> dict[str, Any]:
    paths = list_pdf_paths(state["folder_path"])
    print(f"Found {len(paths)} PDF files")
    return {"pdf_files": paths}


def convert_pdfs(state: PDFProcessorState) -> dict[str, Any]:
    documents = convert_pdf_paths_to_documents(state["pdf_files"])
    print(f"Converted {len(documents)} documents")
    return {"documents": documents}


def chunk_documents(state: PDFProcessorState) -> dict[str, Any]:
    chunked = chunk_document_records(
        state["documents"],
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    print(f"Chunked documents into {len(chunked)} chunks")
    return {"documents": chunked}


def generate_embeddings(state: PDFProcessorState) -> dict[str, Any]:
    texts = [doc["content"] for doc in state["documents"]]
    vectors = embed_texts(texts, EMBEDDING_MODEL)
    print(f"Generated embeddings for {len(vectors)} chunks")
    return {"embeddings": vectors}
