"""LangGraph compile-time wiring and public entrypoints for the PDF workflow."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent_settings import PDF_FOLDER_PATH
from src.graph_nodes import (
    chunk_documents,
    convert_pdfs,
    generate_embeddings,
    load_pdfs,
)
from src.graph_state import PDFProcessorState


def create_pdf_processor_graph():
    """Create the PDF processing workflow graph."""
    workflow = StateGraph(PDFProcessorState)

    workflow.add_node("load_pdfs", load_pdfs)
    workflow.add_node("convert_pdfs", convert_pdfs)
    workflow.add_node("chunk_documents", chunk_documents)
    workflow.add_node("generate_embeddings", generate_embeddings)

    workflow.add_edge(START, "load_pdfs")
    workflow.add_edge("load_pdfs", "convert_pdfs")
    workflow.add_edge("convert_pdfs", "chunk_documents")
    workflow.add_edge("chunk_documents", "generate_embeddings")
    workflow.add_edge("generate_embeddings", END)

    return workflow.compile()


def process_pdfs(
    folder_path: str | None = None,
    *,
    conversion_pipeline: Literal["standard", "vlm"] | None = None,
) -> PDFProcessorState:
    """Run the compiled graph on ``folder_path`` (or the configured default)."""
    workflow = create_pdf_processor_graph()

    initial_state: PDFProcessorState = {
        "folder_path": folder_path or PDF_FOLDER_PATH,
        "pdf_files": [],
        "documents": [],
        "embeddings": [],
    }
    if conversion_pipeline is not None:
        initial_state["conversion_pipeline"] = conversion_pipeline

    return workflow.invoke(initial_state)


if __name__ == "__main__":
    folder_path = PDF_FOLDER_PATH

    print(f"Processing PDFs in folder: {folder_path}")

    try:
        result = process_pdfs(folder_path)
        print("PDF processing completed successfully")
        print(f"Total chunks processed: {len(result['documents'])}")
        print(f"Total embeddings generated: {len(result['embeddings'])}")
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
