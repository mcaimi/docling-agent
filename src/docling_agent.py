#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the document processing workflow."""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from argparse import ArgumentParser

try:
    from .setup import SENSITIVE_SETTING_KEYS, load_environment_variables, settings_contract
    from .utils import configure_logging
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

try:
    from .graph import (
        extract_config_from_prompt,
        list_document_paths,
        merge_overrides,
        shard_paths,
    )
    from .graph.document_paths import DOCUMENT_GLOBS
    from .graph.document_pipeline import document_workflow, progress_metrics
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

logger = logging.getLogger(__name__)

# CLI exit codes. Per-file failures are not fatal: they are reported in
# error_reports and the run still exits 0, so container orchestrators (e.g.
# a Kubernetes Job's backoffLimitPerIndex) only retry genuine crashes.
EXIT_OK = 0
EXIT_FATAL = 1
EXIT_CONFIG_ERROR = 2

# Settings flagged ``sensitive`` in settings_spec are always redacted from
# MLflow params. Any other key that looks like a credential is redacted too --
# a denylist-by-pattern so newly added secret-shaped settings are safe by
# default even if the spec forgets to flag them.
_SENSITIVE_KEY_PATTERN = re.compile(r"(api_key|token|secret|password|headers)$", re.IGNORECASE)


def _is_sensitive_param(key: str) -> bool:
    return key in SENSITIVE_SETTING_KEYS or bool(_SENSITIVE_KEY_PATTERN.search(key))


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
    documents, embeddings, errors, failed_chunks = [], [], [], []
    vectors_stored = 0
    vectors_upserted = 0
    retry_count = 0
    for item in pipeline_result:
        documents.extend(item.get("chunks", []))
        embeddings.extend(item.get("embeddings", []))
        vectors_stored += item.get("stored_count", 0)
        vectors_upserted += item.get("upserted_count", 0)
        retry_count += item.get("retry_count", 0)
        failed_chunks.extend(item.get("failed_chunks", []))
        if item.get("error"):
            errors.append(
                {"path": item["path"], "error": item["error"], "stage": "pipeline"}
            )
        if item.get("embedding_error"):
            errors.append(
                {
                    "path": item["path"],
                    "error": item["embedding_error"],
                    "stage": "embedding",
                }
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
        "vectors_upserted": vectors_upserted,
        "retry_count": retry_count,
        "failed_chunks": failed_chunks,
        "error_reports": errors,
        **{
            k: v
            for k, v in agent_settings.items()
            if k not in ("documents", "embeddings")
        },
    }


def _discover_paths(agent_settings: dict[str, Any]) -> tuple[list[str], int]:
    """List documents in folder_path and keep this instance's shard.

    Returns ``(shard_paths, total_document_count)``.
    """
    paths = list_document_paths(agent_settings.get("folder_path"))
    shard_index = int(agent_settings.get("shard_index") or 0)
    shard_count = int(agent_settings.get("shard_count") or 1)
    selected = shard_paths(paths, shard_index, shard_count)
    if shard_count > 1:
        logger.info(
            " -> Shard %d/%d: %d of %d documents",
            shard_index,
            shard_count,
            len(selected),
            len(paths),
        )
    return selected, len(paths)


def _invoke_pipeline(
    agent_settings: dict[str, Any],
    paths: list[str],
    mlflow_run_id: str | None = None,
) -> list[dict[str, Any]]:
    input_data = [{"file_path": p, "config": agent_settings} for p in paths]
    configurable: dict[str, Any] = {
        "max_concurrency": agent_settings.get("max_concurrency")
    }
    if mlflow_run_id:
        configurable["mlflow_run_id"] = mlflow_run_id
    return document_workflow.invoke(input_data, config={"configurable": configurable})


def _run_workflow(agent_settings: dict[str, Any]) -> dict[str, Any]:
    """Run NLP extraction, discovery, pipeline invoke, and result aggregation."""
    agent_settings, _ = _apply_nlp_overrides(agent_settings)
    paths, _ = _discover_paths(agent_settings)
    pipeline_result = _invoke_pipeline(agent_settings, paths)
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
    # Per-document spans are created explicitly (see document_pipeline.py);
    # LangGraph task autolog is opt-in, see settings_spec.mlflow_langchain_autolog.
    if agent_settings.get("mlflow_langchain_autolog"):
        mlflow.langchain.autolog()
    mlflow.openai.autolog()

    parent_run_id = agent_settings.get("mlflow_parent_run_id")
    shard_index = int(agent_settings.get("shard_index") or 0)
    shard_count = int(agent_settings.get("shard_count") or 1)
    if shard_count > 1:
        run_name = f"{run_name}_shard{shard_index}"
    started = time.monotonic()

    # With a parent run (set by the scheduler), this run is one shard of a
    # logical run: a nested child, sharing the parent's trace session.
    with mlflow.start_run(run_name=run_name, parent_run_id=parent_run_id) as run:
        session_id = (
            agent_settings.get("mlflow_session_id")
            or parent_run_id
            or run.info.run_id
        )
        agent_settings["mlflow_session_id"] = session_id
        # Propagated to every task-level span (see document_pipeline.py, chunking.py,
        # embeddings.py, vector_store.py, via utils.mlflow_tracing.run_context) so
        # spans created in LangGraph's worker threads attach to this run instead of
        # each spawning its own orphan run -- MLflow's active-run tracking is
        # per-thread and doesn't reach those threads.
        agent_settings["mlflow_run_id"] = run.info.run_id
        mlflow.set_tag("mlflow.trace.session", session_id)
        mlflow.set_tags(
            {
                "shard_index": shard_index,
                "shard_count": shard_count,
                **{
                    k: agent_settings[k]
                    for k in ("scheduler_run_id", "scheduler_executor")
                    if agent_settings.get(k)
                },
            }
        )

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
                paths, total_document_count = _discover_paths(agent_settings)
                mlflow.log_metric("document_count", len(paths))
                mlflow.set_tag("total_document_count", total_document_count)

            # Progress metrics (total_chunks, vectors_stored, error_count, ...)
            # are logged by document_workflow after every batch, with
            # step=docs_completed; the last batch's values are the totals.
            with mlflow.start_span("Parallel pipeline"):
                pipeline_result = _invoke_pipeline(
                    agent_settings, paths, mlflow_run_id=run.info.run_id
                )

            with mlflow.start_span("Results aggregation"):
                final_result = _aggregate_pipeline_results(
                    agent_settings, pipeline_result
                )
                errors = final_result["error_reports"]
                if errors:
                    mlflow.log_dict(errors, "error_reports.json")
                mlflow.log_dict(
                    {
                        "status": "completed",
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "document_count": len(paths),
                        "total_document_count": total_document_count,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        **progress_metrics(pipeline_result),
                    },
                    "run_summary.json",
                )

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


def _log_run_summary(result: dict[str, Any]) -> None:
    logger.info("Document processing completed successfully")
    logger.info("Total chunks processed: %d", len(result["documents"]))
    logger.info("Total embeddings generated: %d", len(result["embeddings"]))
    backend = (result.get("vector_store_backend") or "none").lower()
    if backend not in ("", "none"):
        logger.info("Vectors stored in %s: %d", backend, result.get("vectors_stored", 0))
    errors = result.get("error_reports", [])
    if errors:
        logger.warning("Files with errors: %d", len(errors))
        for err in errors:
            stage = err.get("stage", "pipeline")
            logger.warning(
                "  - [%s] %s: %s", stage, err["path"], err["error"],
                extra={"file": Path(err["path"]).name, "stage": stage},
            )


# parse command line arguments
def parse_arguments():
    parser = ArgumentParser(description="Document processing agent")
    parser.add_argument(
        "--envfile",
        type=str,
        default=None,
        help=(
            "Environment file to load before reading configuration from the "
            "environment. Defaults to ./.env when present; containers pass "
            "configuration purely through environment variables."
        ),
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default=None,
        help="The user prompt (overrides USER_PROMPT).",
    )
    parser.add_argument(
        "--print-settings-schema",
        action="store_true",
        help="Print the settings contract (JSON) consumed by launchers and exit.",
    )

    # return the parsed arguments
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.print_settings_schema:
        print(json.dumps(settings_contract(DOCUMENT_GLOBS), indent=2))
        return EXIT_OK

    envfile = args.envfile
    if envfile is None and Path(".env").exists():
        envfile = ".env"

    try:
        agent_settings: dict = load_environment_variables(envfile)
        if args.user_prompt is not None:
            agent_settings["user_prompt"] = args.user_prompt
    except Exception as e:
        configure_logging()
        logger.error("Error loading environment variables: %s", e)
        return EXIT_CONFIG_ERROR

    configure_logging(
        agent_settings.get("log_level", "INFO"),
        agent_settings.get("log_format", "text"),
        static_fields={
            "scheduler_run_id": agent_settings.get("scheduler_run_id"),
            "shard_index": agent_settings.get("shard_index"),
        },
    )

    tracing = bool(agent_settings.get("enable_tracing"))
    if tracing:
        logger.info(" -> Tracing is enabled (enable_tracing=%s)", agent_settings.get("enable_tracing"))
        logger.info(" -> MLflow tracking URI: %s", agent_settings.get("mlflow_tracking_uri"))
        logger.info(" -> MLflow experiment name: %s", agent_settings.get("mlflow_experiment_name"))

    try:
        # fail fast on a bad shard assignment, before any MLflow run exists
        shard_paths([], int(agent_settings["shard_index"]), int(agent_settings["shard_count"]))
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        return EXIT_CONFIG_ERROR

    try:
        result = process_documents(agent_settings, enable_tracing=tracing)
    except ImportError as e:
        logger.error(" X %s", e)
        return EXIT_FATAL
    except Exception:
        logger.exception("Error processing documents")
        return EXIT_FATAL

    if tracing:
        session_id = result.get("mlflow_session_id")
        if session_id:
            logger.info("  MLflow session ID: %s", session_id)
    _log_run_summary(result)
    return EXIT_OK


# main entrypoint
if __name__ == "__main__":
    sys.exit(main())
