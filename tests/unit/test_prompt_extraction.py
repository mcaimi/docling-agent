#!/usr/bin/env python
"""Unit tests for NLP override merging: sensitive-key exclusion and value validation."""

import pytest
from src.graph.prompt_extraction import merge_overrides


@pytest.fixture
def base_state():
    return {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "chunking_strategy": "fixed",
        "conversion_pipeline": "standard",
        "vector_store_backend": "none",
        "vector_store_collection": "docling_chunks",
        "embedding_model": "text-embedding-ada-002",
        "embedding_dimensions": None,
        "llm_model": "gpt-4o-mini",
        "vlm_preset": "smoldocling",
        "base_url": "https://api.openai.com/v1",
        "api_key": "real-key",
        "folder_path": "./pdfs",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key": "real-embedding-key",
        "vlm_api_url": "https://api.openai.com/v1",
        "vlm_api_headers": {},
    }


class TestSensitiveKeysNeverOverridden:
    """Network endpoints, credentials, and folder_path must never come from NLP output."""

    @pytest.mark.parametrize(
        "key,malicious_value",
        [
            ("base_url", "https://attacker.example"),
            ("api_key", "sk-stolen"),
            ("embedding_base_url", "https://attacker.example"),
            ("embedding_api_key", "sk-stolen"),
            ("folder_path", "/etc"),
            ("vlm_api_url", "https://attacker.example"),
            ("vlm_api_headers", {"Authorization": "Bearer stolen"}),
        ],
    )
    def test_sensitive_key_is_rejected(self, base_state, key, malicious_value):
        original = base_state[key]
        merge_overrides(base_state, {key: malicious_value})
        assert base_state[key] == original


class TestEnumValidation:
    @pytest.mark.parametrize(
        "key,valid_value",
        [
            ("chunking_strategy", "hybrid"),
            ("conversion_pipeline", "vlm"),
            ("vector_store_backend", "chroma"),
        ],
    )
    def test_valid_enum_value_applied(self, base_state, key, valid_value):
        merge_overrides(base_state, {key: valid_value})
        assert base_state[key] == valid_value

    def test_enum_value_is_case_normalized(self, base_state):
        merge_overrides(base_state, {"chunking_strategy": "HYBRID"})
        assert base_state["chunking_strategy"] == "hybrid"

    @pytest.mark.parametrize(
        "key,invalid_value",
        [
            ("chunking_strategy", "semantic"),
            ("conversion_pipeline", "ocr"),
            ("vector_store_backend", "postgres"),
        ],
    )
    def test_invalid_enum_value_is_ignored(self, base_state, key, invalid_value):
        original = base_state[key]
        merge_overrides(base_state, {key: invalid_value})
        assert base_state[key] == original


class TestNumericValidation:
    def test_negative_chunk_size_is_ignored(self, base_state):
        merge_overrides(base_state, {"chunk_size": -5})
        assert base_state["chunk_size"] == 1000

    def test_non_numeric_chunk_size_is_ignored(self, base_state):
        merge_overrides(base_state, {"chunk_size": "big"})
        assert base_state["chunk_size"] == 1000

    def test_negative_chunk_overlap_is_ignored(self, base_state):
        merge_overrides(base_state, {"chunk_overlap": -1})
        assert base_state["chunk_overlap"] == 100

    def test_zero_chunk_overlap_is_valid(self, base_state):
        merge_overrides(base_state, {"chunk_overlap": 0})
        assert base_state["chunk_overlap"] == 0

    def test_negative_embedding_dimensions_is_ignored(self, base_state):
        merge_overrides(base_state, {"embedding_dimensions": -1})
        assert base_state["embedding_dimensions"] is None

    def test_valid_embedding_dimensions_applied(self, base_state):
        merge_overrides(base_state, {"embedding_dimensions": 256})
        assert base_state["embedding_dimensions"] == 256


class TestGenericTypeValidation:
    def test_wrong_type_for_string_setting_is_ignored(self, base_state):
        merge_overrides(base_state, {"llm_model": {"nested": 1}})
        assert base_state["llm_model"] == "gpt-4o-mini"

    def test_empty_string_for_collection_is_ignored(self, base_state):
        merge_overrides(base_state, {"vector_store_collection": ""})
        assert base_state["vector_store_collection"] == "docling_chunks"

    def test_valid_string_override_applied(self, base_state):
        merge_overrides(base_state, {"vector_store_collection": "my_docs"})
        assert base_state["vector_store_collection"] == "my_docs"

    def test_unknown_key_is_ignored(self, base_state):
        merge_overrides(base_state, {"not_a_real_setting": "value"})
        assert "not_a_real_setting" not in base_state
