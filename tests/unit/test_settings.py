#!/usr/bin/env python
"""Unit tests for the declarative settings spec and the spec-driven loader."""

import os

import pytest

from src.setup.agent_settings import _get_settings_from_environment, load_environment_variables
from src.setup.settings_spec import SENSITIVE_SETTING_KEYS, SETTINGS_SPEC, settings_contract

# Defaults of the hand-written loader that predates the spec. The spec-driven
# loader must keep returning exactly these for an empty environment.
LEGACY_DEFAULTS = {
    "llm_model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key-here",
    "folder_path": "./pdfs",
    "embedding_model": "text-embedding-ada-002",
    "embedding_dimensions": None,
    "embedding_base_url": "https://api.openai.com/v1",
    "embedding_api_key": "your-api-key-here",
    "conversion_pipeline": "standard",
    "chunking_strategy": "fixed",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "vlm_api_url": "https://api.openai.com/v1",
    "vlm_api_headers": {},
    "vlm_api_params": {},
    "vlm_preset": "smoldocling",
    "max_concurrency": os.cpu_count(),
    "enable_tracing": 0,
    "mlflow_tracking_uri": "http://localhost:5000",
    "mlflow_experiment_name": "Docling Agent",
    "mlflow_session_id": None,
    "vector_store_backend": "none",
    "vector_store_collection": "docling_chunks",
    "chroma_host": None,
    "chroma_port": 8000,
    "chroma_persist_directory": None,
    "milvus_uri": "http://localhost:19530",
    "milvus_token": None,
    "milvus_db_name": "default",
    "vector_store_batch_size": 200,
    "vector_store_write_mode": "insert",
    "vector_store_max_batch_workers": 4,
    "vector_store_retry_attempts": 3,
    "vector_store_retry_backoff_multiplier": 1,
    "vector_store_circuit_breaker_threshold": 5,
    "vector_store_circuit_breaker_timeout": 60,
    "milvus_index_type": "HNSW",
    "milvus_metric_type": "COSINE",
    "milvus_index_params": {},
    "milvus_consistency_level": "Bounded",
    "milvus_partition_name": None,
    "chroma_distance_function": "cosine",
    "chroma_collection_metadata": {},
    "chroma_allow_reset": 0,
}


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every env var the spec reads."""
    for spec in SETTINGS_SPEC:
        for key in (spec.env, *spec.env_fallbacks):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_spec_keys_are_unique():
    keys = [s.key for s in SETTINGS_SPEC]
    envs = [s.env for s in SETTINGS_SPEC]
    assert len(keys) == len(set(keys))
    assert len(envs) == len(set(envs))


def test_defaults_match_legacy_loader(clean_env):
    settings = _get_settings_from_environment()
    for key, expected in LEGACY_DEFAULTS.items():
        assert settings[key] == expected, key


def test_new_settings_defaults(clean_env):
    settings = _get_settings_from_environment()
    assert settings["user_prompt"] == ""
    assert settings["shard_index"] == 0
    assert settings["shard_count"] == 1
    assert settings["log_level"] == "INFO"
    assert settings["log_format"] == "text"
    assert settings["mlflow_parent_run_id"] is None
    assert settings["scheduler_run_id"] is None
    assert settings["mlflow_langchain_autolog"] == 0  # opt-in, see settings_spec
    assert settings["source_backend"] == "local"
    assert settings["s3_bucket"] is None
    assert settings["s3_prefix"] == ""
    assert settings["s3_recursive"] == 0
    assert settings["s3_addressing_style"] == "auto"
    assert settings["s3_verify_ssl"] == 1


def test_s3_settings_fall_back_to_aws_env(clean_env):
    clean_env.setenv("AWS_ACCESS_KEY_ID", "aws-id")
    clean_env.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    clean_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    settings = _get_settings_from_environment()
    assert settings["s3_access_key_id"] == "aws-id"
    assert settings["s3_secret_access_key"] == "aws-secret"
    assert settings["s3_region"] == "eu-west-1"
    clean_env.setenv("S3_ACCESS_KEY_ID", "s3-id")
    clean_env.setenv("AWS_REGION", "us-east-2")
    settings = _get_settings_from_environment()
    assert settings["s3_access_key_id"] == "s3-id"
    assert settings["s3_region"] == "us-east-2"


def test_env_values_are_parsed(clean_env):
    clean_env.setenv("CHUNK_SIZE", " 500 ")
    clean_env.setenv("CHUNK_OVERLAP", "not-a-number")  # invalid int -> default
    clean_env.setenv("FOLDER_PATH", "  /data/in ")
    clean_env.setenv("CHROMA_HOST", "   ")  # blank -> default (None)
    clean_env.setenv("VLM_API_HEADERS", '{"Authorization": "Bearer x"}')
    clean_env.setenv("EMBEDDING_DIMENSIONS", "256")
    settings = _get_settings_from_environment()
    assert settings["chunk_size"] == 500
    assert settings["chunk_overlap"] == 100
    assert settings["folder_path"] == "/data/in"
    assert settings["chroma_host"] is None
    assert settings["vlm_api_headers"] == {"Authorization": "Bearer x"}
    assert settings["embedding_dimensions"] == 256


@pytest.mark.parametrize(
    "raw, expected",
    [("1", 1), ("true", 1), ("YES", 1), ("on", 1), ("0", 0), ("false", 0), ("off", 0),
     ("2", 1), ("maybe", 0)],
)
def test_bool_settings(clean_env, raw, expected):
    clean_env.setenv("ENABLE_TRACING", raw)
    assert _get_settings_from_environment()["enable_tracing"] == expected


def test_invalid_json_raises(clean_env):
    clean_env.setenv("VLM_API_PARAMS", "[1, 2]")
    with pytest.raises(ValueError):
        _get_settings_from_environment()


def test_shard_index_falls_back_to_job_completion_index(clean_env):
    clean_env.setenv("JOB_COMPLETION_INDEX", "3")
    assert _get_settings_from_environment()["shard_index"] == 3
    clean_env.setenv("SHARD_INDEX", "1")
    assert _get_settings_from_environment()["shard_index"] == 1


def test_env_only_loading_without_envfile(clean_env):
    clean_env.setenv("CHUNKING_STRATEGY", "hybrid")
    assert load_environment_variables()["chunking_strategy"] == "hybrid"


def test_envfile_is_loaded(clean_env, tmp_path):
    envfile = tmp_path / "test.env"
    envfile.write_text("CHUNK_SIZE=321\n")
    try:
        assert load_environment_variables(str(envfile))["chunk_size"] == 321
    finally:
        # load_dotenv writes straight into os.environ, outside monkeypatch's reach
        os.environ.pop("CHUNK_SIZE", None)


def test_missing_envfile_raises(clean_env, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_environment_variables(str(tmp_path / "missing.env"))


def test_sensitive_keys():
    assert SENSITIVE_SETTING_KEYS == {
        "api_key", "embedding_api_key", "vlm_api_headers", "vlm_api_params", "milvus_token",
        "s3_access_key_id", "s3_secret_access_key", "s3_session_token",
    }


def test_contract_never_exposes_sensitive_defaults():
    schema = {entry["key"]: entry for entry in settings_contract(["*.md"])["settings"]}
    assert schema.keys() == {s.key for s in SETTINGS_SPEC}
    for key in SENSITIVE_SETTING_KEYS:
        assert schema[key]["default"] is None
    assert schema["user_prompt"]["untrusted"] is True
    assert schema["chunk_size"]["depends_on"] == {"chunking_strategy": ["fixed"]}


def test_contract_names_computed_defaults():
    contract = settings_contract(["*.md"])
    schema = {entry["key"]: entry for entry in contract["settings"]}
    assert schema["max_concurrency"]["default"] is None
    assert schema["max_concurrency"]["dynamic_default"] == "cpu_count"
    assert schema["chunk_size"]["dynamic_default"] is None
    assert contract["document_globs"] == ["*.md"]
