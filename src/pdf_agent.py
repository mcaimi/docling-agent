#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the PDF workflow."""

from typing import Any
from argparse import ArgumentParser

# import langgraph
try:
    from langgraph.graph import END, START, StateGraph
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# import the agent settings
try:
    from .setup import load_environment_variables
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
    agent_settings: dict[str, Any],
) -> PDFProcessorState:
    """Run the compiled graph on ``folder_path`` (or the configured default)."""
    workflow = create_pdf_processor_graph()

    initial_state: PDFProcessorState = {
        **agent_settings,
    }

    return workflow.invoke(initial_state)

# parse command line arguments
def parse_arguments():
    parser = ArgumentParser(description="PDF processing agent")
    parser.add_argument(
        "--envfile", 
        type=str, 
        default=".env", 
        help="Environment file to use as configuration."
    )
    
    # return the parsed arguments
    return parser.parse_args()

# main entrypoint
if __name__ == "__main__":
    # parse command line arguments
    args = parse_arguments()
    envfile = args.envfile

    # load environment variables
    try:
        agent_settings = load_environment_variables(envfile)
    except Exception as e:
        print(f"Error loading environment variables: {str(e)}")
        exit(1)

    try:
        result = process_pdfs(agent_settings)
        print("PDF processing completed successfully")
        print(f"Total chunks processed: {len(result['documents'])}")
        print(f"Total embeddings generated: {len(result['embeddings'])}")
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
