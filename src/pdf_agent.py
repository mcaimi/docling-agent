#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the PDF workflow."""

from typing import Literal

# import langgraph
try:
    from langgraph.graph import END, START, StateGraph
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# import the agent settings
try:
    from .setup import agent_settings
    from .utils import _parse_json_object_str
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# import the graph nodes and processor state
try:
    from .graph import (
        chunk_documents,
        convert_pdfs,
        generate_embeddings,
        load_pdfs,
        PDFProcessorState,
    )
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# create the PDF processing workflow graph
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


# process documents
def process_pdfs(
    folder_path: str | None = None,
    *,
    conversion_pipeline: Literal["standard", "vlm"] | None = None,
) -> PDFProcessorState:
    """Run the compiled graph on ``folder_path`` (or the configured default)."""
    workflow = create_pdf_processor_graph()

    initial_state: PDFProcessorState = {
        "folder_path": folder_path or PDF_FOLDER_PATH,
        "api_key": agent_settings.EMBEDDING_API_KEY or agent_settings.OPENAI_API_KEY,
        "chunk_size": agent_settings.CHUNK_SIZE,
        "chunk_overlap": agent_settings.CHUNK_OVERLAP,
        "vlm_api_url": agent_settings.DOCLING_VLM_API_URL_ENV,
        "conversion_pipeline": agent_settings.DOCLING_PDF_PIPELINE_ENV,
        "embedding_model": agent_settings.EMBEDDING_MODEL,
        "connection_params": {
            "api_key": agent_settings.EMBEDDING_API_KEY or agent_settings.OPENAI_API_KEY,
            "base_url": agent_settings.EMBEDDING_BASE_URL,
        },
        "vlm_api_headers": _parse_json_object_str(agent_settings.DOCLING_VLM_API_HEADERS_ENV),
        "vlm_api_params": _parse_json_object_str(agent_settings.DOCLING_VLM_API_PARAMS_ENV),
        "vlm_preset": agent_settings.DOCLING_VLM_PRESET_ENV,
        "pdf_files": [],
        "documents": [],
        "embeddings": [],
    }
    if conversion_pipeline is not None:
        initial_state["conversion_pipeline"] = conversion_pipeline

    return workflow.invoke(initial_state)


# main entrypoint
if __name__ == "__main__":
    folder_path = agent_settings.PDF_FOLDER_PATH

    print(f"Processing PDFs in folder: {folder_path}")

    try:
        result = process_pdfs(folder_path)
        print("PDF processing completed successfully")
        print(f"Total chunks processed: {len(result['documents'])}")
        print(f"Total embeddings generated: {len(result['embeddings'])}")
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
