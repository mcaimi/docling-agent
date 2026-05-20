#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the PDF workflow."""

import os
from typing import Any
from argparse import ArgumentParser

try:
    from .setup import load_environment_variables
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

try:
    from .graph import extract_config_from_prompt, list_pdf_paths, merge_overrides
    from .graph.document_pipeline import document_workflow
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e


def process_pdfs(
    agent_settings: dict[str, Any],
) -> dict[str, Any]:
    """Run the PDF processing workflow on all PDFs in folder_path with per-document parallelism."""
    # 1. NLP extraction (synchronous)
    user_prompt = agent_settings.get("user_prompt", "")
    # process the user prompt if requested
    config: dict = {}
    if user_prompt:
        config = extract_config_from_prompt(
            user_prompt,
            agent_settings.get("llm_model"),
            agent_settings.get("base_url"),
            agent_settings.get("api_key"),
        )
    filtered = {k: v for k, v in config.items() if v is not None}
    # update the graph state
    agent_settings = merge_overrides(agent_settings, filtered)

    # 2. List PDFs
    paths = list_pdf_paths(agent_settings.get("folder_path"))

    # 3. Determine concurrency level (if not specified, use the number of CPUs available, if unable to determine use 1)
    max_conc = agent_settings.get("max_concurrency")

    # 4. Build input for the pipeline: list of {"pdf_path", "config"} dicts
    input_data = [{"pdf_path": p, "config": agent_settings} for p in paths]

    # 5. Run parallel pipeline
    result = document_workflow.invoke(
        input_data,
        config={"configurable": {"max_concurrency": max_conc}},
    )

    # 6. Gather results
    documents, embeddings, errors = [], [], []
    for item in result:
        documents.extend(item["chunks"])
        embeddings.extend(item["embeddings"])
        if item.get("error"):
            errors.append({"path": item["path"], "error": item["error"]})

    return {
        "documents": documents,
        "embeddings": embeddings,
        "error_reports": errors,
        **{
            k: v
            for k, v in agent_settings.items()
            if k not in ("documents", "embeddings")
        },
    }


# parse command line arguments
def parse_arguments():
    parser = ArgumentParser(description="PDF processing agent")
    parser.add_argument(
        "--envfile",
        type=str,
        default=".env",
        help="Environment file to use as configuration.",
    )
    parser.add_argument("--user_prompt", type=str, default="", help="The user prompt")

    # return the parsed arguments
    return parser.parse_args()


# main entrypoint
if __name__ == "__main__":
    # parse command line arguments
    args = parse_arguments()
    envfile = args.envfile

    # load environment variables
    try:
        agent_settings: dict = load_environment_variables(envfile)
        agent_settings["user_prompt"] = args.user_prompt
    except Exception as e:
        print(f"Error loading environment variables: {str(e)}")
        exit(1)

    try:
        result = process_pdfs(agent_settings)
        print("Document processing completed successfully")
        print(f"Total chunks processed: {len(result['documents'])}")
        print(f"Total embeddings generated: {len(result['embeddings'])}")
        errors = result.get("error_reports", [])
        if errors:
            print(f"Files with errors: {len(errors)}")
            for e in errors:
                print(f"  - {e['path']}: {e['error']}")
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        raise e
