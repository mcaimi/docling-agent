#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the document processing workflow."""

import re
import sys
from typing import Any
from argparse import ArgumentParser

try:
    from .setup import load_environment_variables
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

try:
    from .graph import extract_config_from_prompt, list_document_paths, merge_overrides
    from .graph.document_pipeline import document_workflow
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

# Explicit names kept for clarity/documentation; matched as a fallback alongside
# the pattern below so a key doesn't need to be enumerated here to be redacted.
_SENSITIVE_MLFLOW_PARAMS = {
    "api_key",
    "embedding_api_key",
    "vlm_api_headers",
    "vlm_api_params",
    "milvus_token",
}

# Any config key that looks like a credential is redacted from MLflow params,
# regardless of whether it's been explicitly enumerated above. This is a
# denylist-by-pattern so newly added secret-shaped settings are safe by default.
_SENSITIVE_KEY_PATTERN = re.compile(r"(api_key|token|secret|password|headers)$", re.IGNORECASE)


def _is_sensitive_param(key: str) -> bool:
    return key in _SENSITIVE_MLFLOW_PARAMS or bool(_SENSITIVE_KEY_PATTERN.search(key))


def _apply_nlp_overrides(agent_settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract config from user_prompt when set; return updated settings and applied overrides."""
    user_prompt = agent_settings.get("user_prompt", "")
    config: dict = {}
    if user_prompt:
        config = extract_config_from_prompt(
            user_prompt,
            agent_settings.get("llm_model"),
            agent_settings.get("base_url"),
            agent_settings.get("api_key"),
        )
    filtered = {k: v for k, v in config.items() if v is not None}
    return merge_overrides(agent_settings, filtered), filtered


def _aggregate_pipeline_results(
    agent_settings: dict[str, Any],
    pipeline_result: list[dict[str, Any]],
) -> dict[str, Any]:
    documents, embeddings, errors = [], [], []
    vectors_stored = 0
    for item in pipeline_result:
        documents.extend(item.get("chunks", []))
        embeddings.extend(item.get("embeddings", []))
        vectors_stored += item.get("stored_count", 0)
        if item.get("error"):
            errors.append(
                {"path": item["path"], "error": item["error"], "stage": "pipeline"}
            )
        if item.get("store_error"):
            errors.append(
                {
                    "path": item["path"],
                    "error": item["store_error"],
                    "stage": "vector_store",
                }
            )

    return {
        "documents": documents,
        "embeddings": embeddings,
        "vectors_stored": vectors_stored,
        "error_reports": errors,
        **{
            k: v
            for k, v in agent_settings.items()
            if k not in ("documents", "embeddings")
        },
    }


def _run_workflow(agent_settings: dict[str, Any]) -> dict[str, Any]:
    """Run NLP extraction, discovery, pipeline invoke, and result aggregation."""
    agent_settings, _ = _apply_nlp_overrides(agent_settings)
    paths = list_document_paths(agent_settings.get("folder_path"))
    max_conc = agent_settings.get("max_concurrency")
    input_data = [{"file_path": p, "config": agent_settings} for p in paths]
    pipeline_result = document_workflow.invoke(
        input_data,
        config={"configurable": {"max_concurrency": max_conc}},
    )
    return _aggregate_pipeline_results(agent_settings, pipeline_result)


def _run_workflow_with_mlflow(agent_settings: dict[str, Any]) -> dict[str, Any]:
    """Run the workflow inside an MLflow run with top-level spans and metrics."""
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            f"Cannot import mlflow: {e}. Tracing requires mlflow to be installed."
        ) from e

    from .utils import generate_mlflow_run_name, session_context

    run_name = generate_mlflow_run_name()
    mlflow.set_tracking_uri(agent_settings.get("mlflow_tracking_uri"))
    mlflow.set_experiment(agent_settings.get("mlflow_experiment_name"))
    mlflow.langchain.autolog()
    mlflow.openai.autolog()

    with mlflow.start_run(run_name=run_name) as run:
        session_id = agent_settings.get("mlflow_session_id") or run.info.run_id
        agent_settings["mlflow_session_id"] = session_id
        mlflow.set_tag("mlflow.trace.session", session_id)

        sanitised = {
            k: v for k, v in agent_settings.items() if not _is_sensitive_param(k)
        }
        mlflow.log_params(sanitised)
        mlflow.set_tag(
            "nlp_extracted_keys", agent_settings.get("_nlp_extracted_keys", "none")
        )

        with session_context(session_id):
            with mlflow.start_span("NLP prompt extraction"):
                agent_settings, filtered = _apply_nlp_overrides(agent_settings)
                if filtered:
                    mlflow.set_tag(
                        "nlp_extracted_keys", ", ".join(sorted(filtered.keys()))
                    )

            with mlflow.start_span("document discovery"):
                paths = list_document_paths(agent_settings.get("folder_path"))
                mlflow.log_metric("document_count", len(paths))

            max_conc = agent_settings.get("max_concurrency")
            input_data = [{"file_path": p, "config": agent_settings} for p in paths]

            with mlflow.start_span("Parallel pipeline"):
                pipeline_result = document_workflow.invoke(
                    input_data,
                    config={"configurable": {"max_concurrency": max_conc}},
                )

            with mlflow.start_span("Results aggregation"):
                final_result = _aggregate_pipeline_results(
                    agent_settings, pipeline_result
                )
                documents = final_result["documents"]
                embeddings = final_result["embeddings"]
                errors = final_result["error_reports"]
                vectors_stored = final_result["vectors_stored"]

                mlflow.log_metric("total_chunks", len(documents))
                mlflow.log_metric("total_embeddings", len(embeddings))
                mlflow.log_metric("vectors_stored", vectors_stored)
                mlflow.log_metric("error_count", len(errors))
                mlflow.log_metric(
                    "vector_store_error_count",
                    sum(1 for e in errors if e.get("stage") == "vector_store"),
                )
                mlflow.log_metric(
                    "non_null_embeddings",
                    sum(1 for e in embeddings if e is not None),
                )
                if errors:
                    mlflow.log_dict(errors, "error_reports.json")

                return final_result


def process_documents(
    agent_settings: dict[str, Any],
    *,
    enable_tracing: bool = False,
) -> dict[str, Any]:
    """Run the document processing workflow on all documents in folder_path with per-document parallelism.

    When ``enable_tracing`` is true, wraps the workflow in an MLflow run with spans, metrics,
    and optional error artifact logging. Task-level spans in pipeline modules still require
    ``ENABLE_TRACING=1`` in the environment before those modules are imported.
    """
    if enable_tracing:
        return _run_workflow_with_mlflow(agent_settings)
    return _run_workflow(agent_settings)


def _print_run_summary(result: dict[str, Any]) -> None:
    print("Document processing completed successfully")
    print(f"Total chunks processed: {len(result['documents'])}")
    print(f"Total embeddings generated: {len(result['embeddings'])}")
    backend = (result.get("vector_store_backend") or "none").lower()
    if backend not in ("", "none"):
        print(f"Vectors stored in {backend}: {result.get('vectors_stored', 0)}")
    errors = result.get("error_reports", [])
    if errors:
        print(f"Files with errors: {len(errors)}")
        for err in errors:
            stage = err.get("stage", "pipeline")
            print(f"  - [{stage}] {err['path']}: {err['error']}")


# parse command line arguments
def parse_arguments():
    parser = ArgumentParser(description="Document processing agent")
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
    args = parse_arguments()
    envfile = args.envfile

    try:
        agent_settings: dict = load_environment_variables(envfile)
        agent_settings["user_prompt"] = args.user_prompt
    except Exception as e:
        print(f"Error loading environment variables: {str(e)}")
        exit(1)

    tracing = bool(agent_settings.get("enable_tracing"))
    if tracing:
        print(f" -> Tracing is enabled (enable_tracing={agent_settings.get('enable_tracing')})")
        print(f" -> MLflow tracking URI: {agent_settings.get('mlflow_tracking_uri')}")
        print(f" -> MLflow experiment name: {agent_settings.get('mlflow_experiment_name')}")

    try:
        result = process_documents(agent_settings, enable_tracing=tracing)
        if tracing:
            session_id = result.get("mlflow_session_id")
            if session_id:
                print(f"  MLflow session ID: {session_id}")
        _print_run_summary(result)
    except ImportError as e:
        print(f" X {e}")
        sys.exit(-1)
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise
