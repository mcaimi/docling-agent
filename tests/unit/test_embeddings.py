#!/usr/bin/env python
"""Unit tests for embedding dimension wiring."""

from unittest.mock import MagicMock

import src.graph.embeddings as embeddings_mod


def _fake_client(captured_kwargs: list[dict]) -> MagicMock:
    client = MagicMock()

    def _create(**kwargs):
        captured_kwargs.append(kwargs)
        item = MagicMock()
        item.index = 0
        item.embedding = [0.1, 0.2, 0.3]
        response = MagicMock()
        response.data = [item]
        return response

    client.embeddings.create.side_effect = _create
    return client


def test_dimensions_omitted_when_not_set(monkeypatch):
    captured = []
    client = _fake_client(captured)
    monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

    embeddings_mod.embed_texts(["hello"], "text-embedding-ada-002", "http://x", "key")

    assert "dimensions" not in captured[0]


def test_dimensions_passed_when_set(monkeypatch):
    captured = []
    client = _fake_client(captured)
    monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

    embeddings_mod.embed_texts(
        ["hello"], "text-embedding-3-small", "http://x", "key", dimensions=256
    )

    assert captured[0]["dimensions"] == 256
