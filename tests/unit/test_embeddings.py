#!/usr/bin/env python
"""Unit tests for embedding dimension wiring and client cache concurrency."""

import threading
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

    vectors, errors = embeddings_mod.embed_texts(
        ["hello"], "text-embedding-ada-002", "http://x", "key"
    )

    assert "dimensions" not in captured[0]
    assert errors == []
    assert vectors == [[0.1, 0.2, 0.3]]


def test_dimensions_passed_when_set(monkeypatch):
    captured = []
    client = _fake_client(captured)
    monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

    vectors, errors = embeddings_mod.embed_texts(
        ["hello"], "text-embedding-3-small", "http://x", "key", dimensions=256
    )

    assert captured[0]["dimensions"] == 256
    assert errors == []


class TestEmbeddingRetry:
    """Transient failures should be retried before a batch is given up on."""

    def test_transient_error_is_retried_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(embeddings_mod, "wait_exponential", lambda **kw: lambda *a: 0)
        call_count = {"n": 0}

        def _create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise embeddings_mod.RateLimitError(
                    "rate limited", response=MagicMock(), body=None
                )
            item = MagicMock()
            item.index = 0
            item.embedding = [0.9]
            response = MagicMock()
            response.data = [item]
            return response

        client = MagicMock()
        client.embeddings.create.side_effect = _create
        monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

        vectors, errors = embeddings_mod.embed_texts(["hello"], "model", "http://x", "key")

        assert call_count["n"] == 3
        assert vectors == [[0.9]]
        assert errors == []

    def test_retries_exhausted_reports_error_without_raising(self, monkeypatch):
        def _create(**kwargs):
            raise embeddings_mod.RateLimitError("still limited", response=MagicMock(), body=None)

        client = MagicMock()
        client.embeddings.create.side_effect = _create
        monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

        vectors, errors = embeddings_mod.embed_texts(
            ["a", "b"], "model", "http://x", "key"
        )

        assert vectors == [None, None]
        assert len(errors) == 1
        assert "still limited" in errors[0]
        assert client.embeddings.create.call_count == embeddings_mod._EMBEDDING_RETRY_ATTEMPTS

    def test_non_retryable_error_fails_immediately(self, monkeypatch):
        call_count = {"n": 0}

        def _create(**kwargs):
            call_count["n"] += 1
            raise ValueError("bad request")

        client = MagicMock()
        client.embeddings.create.side_effect = _create
        monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *_: client)

        vectors, errors = embeddings_mod.embed_texts(["a"], "model", "http://x", "key")

        assert call_count["n"] == 1
        assert vectors == [None]
        assert len(errors) == 1


class TestEmbeddingClientCacheConcurrency:
    """get_embedding_client is called concurrently by every document-pipeline
    task in a batch; distinct (base_url, api_key) pairs must never cross-
    contaminate each other's cached client, and each distinct pair should only
    construct one client even when many threads request it at once.
    """

    def test_concurrent_calls_with_different_configs_dont_cross_contaminate(
        self, monkeypatch
    ):
        embeddings_mod._embedding_clients.clear()
        construction_count: dict[tuple, int] = {}
        lock = threading.Lock()

        class FakeClient:
            def __init__(self, cache_key):
                self.cache_key = cache_key

        def fake_openai(api_key, base_url, http_client):
            key = (base_url, api_key)
            with lock:
                construction_count[key] = construction_count.get(key, 0) + 1
            return FakeClient(key)

        monkeypatch.setattr(embeddings_mod, "OpenAI", fake_openai)
        monkeypatch.setattr(embeddings_mod.httpx2, "Client", lambda **kw: object())

        n_configs = 10
        calls_per_config = 50
        results: list[tuple] = []
        results_lock = threading.Lock()

        def worker(config_id: int):
            base_url = f"https://provider-{config_id}.example/v1"
            api_key = f"key-{config_id}"
            for _ in range(calls_per_config):
                client = embeddings_mod.get_embedding_client(base_url, api_key)
                with results_lock:
                    results.append((config_id, client.cache_key))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_configs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # every call for a given config_id must have gotten the client for its
        # own (base_url, api_key) -- never another thread's
        for config_id, cache_key in results:
            expected = (f"https://provider-{config_id}.example/v1", f"key-{config_id}")
            assert cache_key == expected, f"cross-contamination: {config_id} got {cache_key}"

        # each distinct config should have been constructed exactly once,
        # regardless of how many concurrent callers requested it
        assert all(count == 1 for count in construction_count.values()), construction_count
        assert len(construction_count) == n_configs
