#!/usr/bin/env python
"""LangGraph compile-time wiring and public entrypoints for the document processing workflow."""

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


def process_documents(
    agent_settings: dict[str, Any],
) -> dict[str, Any]:
    """Run the document processing workflow on all documents in folder_path with per-document parallelism."""
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

    # 2. List documents
    paths = list_document_paths(agent_settings.get("folder_path"))

    # 3. Determine concurrency level (if not specified, use the number of CPUs available, if unable to determine use 1)
    max_conc = agent_settings.get("max_concurrency")

    # 4. Build input for the pipeline: list of {"file_path", "config"} dicts
    input_data = [{"file_path": p, "config": agent_settings} for p in paths]

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

    # enable tracing
    if agent_settings.get("enable_tracing"):
        try:
            print(
                f" -> Tracing is enabled (enable_tracing={agent_settings.get('enable_tracing')})"
            )
            import mlflow
            from .utils import generate_mlflow_run_name, session_context

            _run_name = generate_mlflow_run_name()
            print(f" -> MLflow run name: {_run_name}")
            print(f" -> MLflow tracking URI: {agent_settings.get('mlflow_tracking_uri')}")
            print(f" -> MLflow experiment name: {agent_settings.get('mlflow_experiment_name')}")

            mlflow.set_tracking_uri(agent_settings.get("mlflow_tracking_uri"))
            mlflow.set_experiment(agent_settings.get("mlflow_experiment_name"))
            mlflow.langchain.autolog()
            mlflow.openai.autolog()

            # instrumented code path
            with mlflow.start_run(run_name=_run_name) as run:
                session_id = agent_settings.get("mlflow_session_id") or run.info.run_id
                agent_settings["mlflow_session_id"] = session_id
                mlflow.set_tag("mlflow.trace.session", session_id)
                print(f"  MLflow session ID: {session_id}")

                # Log all agent settings as params, filtering sensitive data
                _sensitive = {"api_key", "embedding_api_key", "vlm_api_headers", "vlm_api_params"}
                _sanitised = {
                    k: v for k, v in agent_settings.items() if k not in _sensitive
                }
                mlflow.log_params(_sanitised)
                print(f"  MLflow Run ID: {run.info.run_id}")
                mlflow.set_tag("nlp_extracted_keys", agent_settings.get("_nlp_extracted_keys", "none"))

                with session_context(session_id):
                    # Step 1: NLP prompt extraction
                    with mlflow.start_span("NLP prompt extraction"):
                        _user_prompt = agent_settings.get("user_prompt", "")
                        _cfg: dict = {}
                        if _user_prompt:
                            _cfg = extract_config_from_prompt(
                                _user_prompt,
                                agent_settings.get("llm_model"),
                                agent_settings.get("base_url"),
                                agent_settings.get("api_key"),
                            )
                        _filtered = {k: v for k, v in _cfg.items() if v is not None}
                        agent_settings = merge_overrides(agent_settings, _filtered)
                        if _filtered:
                            mlflow.set_tag(
                                "nlp_extracted_keys", ", ".join(sorted(_filtered.keys()))
                            )
                            print(f"    NLP extracted overrides: {_filtered}")

                    # Step 2: document discovery
                    with mlflow.start_span("document discovery"):
                        _paths = list_document_paths(agent_settings.get("folder_path"))
                        mlflow.log_metric("document_count", len(_paths))

                    # Step 3: Build input
                    _max_conc = agent_settings.get("max_concurrency")
                    _input_data = [
                        {"file_path": p, "config": agent_settings} for p in _paths
                    ]

                    # Step 4: Pipeline invoke
                    with mlflow.start_span("Parallel pipeline"):
                        _result = document_workflow.invoke(
                            _input_data,
                            config={"configurable": {"max_concurrency": _max_conc}},
                        )

                    # Step 5: Results aggregation
                    with mlflow.start_span("Results aggregation"):
                        _docs, _embs, _errs = [], [], []
                        for item in _result:
                            _docs.extend(item["chunks"])
                            _embs.extend(item["embeddings"])
                            if item.get("error"):
                                _errs.append({"path": item["path"], "error": item["error"]})

                        _final_result = {
                            "documents": _docs,
                            "embeddings": _embs,
                            "error_reports": _errs,
                            **{
                                k: v
                                for k, v in agent_settings.items()
                                if k not in ("documents", "embeddings")
                            },
                        }

                        # Log metrics
                        mlflow.log_metric("total_chunks", len(_docs))
                        mlflow.log_metric("total_embeddings", len(_embs))
                        mlflow.log_metric("error_count", len(_errs))
                        mlflow.log_metric(
                            "non_null_embeddings",
                            sum(1 for e in _embs if e is not None),
                        )

                        # Log error reports as artifact
                        if _errs:
                            mlflow.log_dict(_errs, "error_reports.json")

                        print("Document processing completed successfully")
                        print(f"Total chunks processed: {len(_docs)}")
                        print(f"Total embeddings generated: {len(_embs)}")
                        if _errs:
                            print(f"Files with errors: {len(_errs)}")
                            for e in _errs:
                                print(f"  - {e['path']}: {e['error']}")
                    result = _final_result
        except ImportError as e:
            print(f" X Cannot import mlflow: {e}. Auto Tracing Cannot Start.")
            sys.exit(-1)
    else:
        try:
            result = process_documents(agent_settings)
            print("Document processing completed successfully")
            print(f"Total chunks processed: {len(result['documents'])}")
            print(f"Total embeddings generated: {len(result['embeddings'])}")
            errors = result.get("error_reports", [])
            if errors:
                print(f"Files with errors: {len(errors)}")
                for e in errors:
                    print(f"  - {e['path']}: {e['error']}")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise e
