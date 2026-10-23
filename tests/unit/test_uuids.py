#!/usr/bin/env python
"""Unit tests for UUID helpers."""

from src.utils.uuids import generate_uuid, is_valid_uuid


class TestIsValidUuid:
    def test_generated_uuid_is_valid(self):
        assert is_valid_uuid(generate_uuid()) is True

    def test_known_valid_uuid(self):
        assert is_valid_uuid("123e4567-e89b-12d3-a456-426614174000") is True

    def test_garbage_string_is_invalid(self):
        assert is_valid_uuid("not-a-uuid") is False

    def test_empty_string_is_invalid(self):
        assert is_valid_uuid("") is False
