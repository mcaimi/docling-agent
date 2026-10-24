#!/usr/bin/env python
"""OpenAI-compatible embedding client and batch embedding helpers."""

# import the openai client
try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    # openai>=3 moved its HTTP transport to httpx2 (OpenAI.__init__'s
    # http_client parameter is typed httpx2.Client, not the classic httpx's).
    # A classic httpx.Client happens to still be accepted today via an
    # undocumented compatibility path, but that's not the supported API.
    import httpx2
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError as e:
    raise ImportError(f"tenacity is required: {e}") from e

import os
import threading
from typing import Any

from ..utils.mlflow_tracing import run_context

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# OpenAI embeddings API batch size limit
# Reference: https://platform.openai.com/docs/api-reference/embeddings
_EMBEDDING_BATCH_SIZE = 2048

# Transient errors worth retrying (rate limits, timeouts, connection issues,
# upstream 5xx). Anything else (bad request, auth failure, etc.) fails
# immediately -- retrying those would just waste time and hide the real error.
_RETRYABLE_EMBEDDING_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
# Mirrors the defaults used for vector store writes (see vector_store.py's
# _insert_batch_with_retry) for consistency across the pipeline's resilience behavior.
_EMBEDDING_RETRY_ATTEMPTS = 3
_EMBEDDING_RETRY_BACKOFF_MULTIPLIER = 1

# Embedding clients cache, keyed by (base_url, api_key) so that concurrent
# document-pipeline tasks running with different endpoints/credentials (e.g.
# separate requests in a long-lived service) each get their own client instead
# of racing to overwrite a single shared slot. Guarded by a lock: the previous
# single-slot cache had a check-then-set race where two threads with different
# configs could interleave and hand one thread the other's client/credentials.
_embedding_clients: dict[tuple[str, str], OpenAI] = {}
_embedding_clients_lock = threading.Lock()


# get the embedding client
# uses the connection params to configure the client
# connection settings are stored in the graph state
def get_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Return a cached OpenAI client with connection pooling, keyed by (base_url, api_key)."""
    cache_key = (base_url, api_key)

    client = _embedding_clients.get(cache_key)
    if client is not None:
        return client

    with _embedding_clients_lock:
        # re-check: another thread may have created it while we waited for the lock
        client = _embedding_clients.get(cache_key)
        if client is not None:
            return client

        # Configure httpx2 client with connection pooling
        http_client = httpx2.Client(
            limits=httpx2.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            timeout=httpx2.Timeout(60.0, connect=10.0),
        )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        _embedding_clients[cache_key] = client
        return client


def _embed_batch(
    client: OpenAI, texts: list[str], model: str, dimensions: int | None = None
) -> tuple[list[list[float] | None], str | None]:
    """
    Embed a single batch of texts (up to _EMBEDDING_BATCH_SIZE items).

    Transient errors (rate limits, timeouts, connection issues, upstream 5xx)
    are retried with exponential backoff before giving up. Returns embeddings
    in the same order as input texts; if the batch ultimately fails, all
    positions are ``None`` and the second tuple element holds the error
    message (``None`` on success) so callers can surface it instead of
    silently dropping the chunks.
    """
    kwargs: dict[str, Any] = {"input": texts, "model": model}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    @retry(
        stop=stop_after_attempt(_EMBEDDING_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=_EMBEDDING_RETRY_BACKOFF_MULTIPLIER, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE_EMBEDDING_ERRORS),
        reraise=True,
    )
    def _do_create():
        return client.embeddings.create(**kwargs)

    try:
        response = _do_create()
        # Sort by index to ensure correct order (API may return out of order)
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_embeddings], None
    except Exception as e:
        error = f"Error generating embeddings for batch of {len(texts)} texts: {e}"
        print(error)
        return [None] * len(texts), error


# embed the texts
# uses the embedding client to generate embeddings for the texts
# returns a list of embeddings
def embed_texts(
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str,
    dimensions: int | None = None,
    run_id: str | None = None,
) -> tuple[list[list[float] | None], list[str]]:
    """
    Request embeddings for all texts using batched API calls.

    Splits input into batches of up to 2048 items (OpenAI API limit) and makes
    parallel-capable requests. Transient errors (rate limits, timeouts,
    connection issues, upstream 5xx) are retried with backoff before a batch
    is considered failed. For a batch that still fails after retries, those
    positions are ``None`` and the error is recorded in the returned error
    list, so callers can surface it instead of it only being printed.

    ``dimensions``, when set, is passed through to the provider's embeddings
    request. Only a subset of embedding models (e.g. OpenAI's
    ``text-embedding-3-*`` family) accept this parameter -- leave it unset
    (the default) for models that don't.

    ``run_id``, when set, re-enters that run as active before creating this
    call's MLflow span (see ``utils.mlflow_tracing.run_context``). Needed
    because this is typically called from a LangGraph task worker thread, whose
    context doesn't inherit the active run from the thread that started it --
    without it, MLflow would spawn a new orphan run to hold the span instead.

    Returns a ``(embeddings, errors)`` tuple. ``embeddings`` is in the same
    order as input texts; ``errors`` holds one message per failed batch (empty
    if every batch succeeded).
    """
    if not texts:
        return [], []

    client = get_embedding_client(base_url, api_key)
    all_embeddings: list[list[float] | None] = []
    errors: list[str] = []
    embedding_dimensions: int | None = None

    # Process in batches
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        batch_embeddings, batch_error = _embed_batch(client, batch, model, dimensions)
        all_embeddings.extend(batch_embeddings)
        if batch_error is not None:
            errors.append(batch_error)

        # Capture dimensions from first successful embedding
        if embedding_dimensions is None:
            for emb in batch_embeddings:
                if emb is not None:
                    embedding_dimensions = len(emb)
                    break

    if enable_tracing:
        with run_context(run_id), _ml.start_span("embed_texts"):
            _ml.set_tag("model", model)
            _ml.set_tag("text_count", len(texts))
            _ml.set_tag("batch_count", (len(texts) + _EMBEDDING_BATCH_SIZE - 1) // _EMBEDDING_BATCH_SIZE)
            _ml.set_tag("batch_size", _EMBEDDING_BATCH_SIZE)
            if embedding_dimensions is not None:
                _ml.set_tag("embedding_dimensions", embedding_dimensions)
            _ml.set_tag("success_count", sum(1 for e in all_embeddings if e is not None))
            _ml.set_tag("failure_count", sum(1 for e in all_embeddings if e is None))

    return all_embeddings, errors


def embed_single_text(
    text: str,
    model: str,
    base_url: str,
    api_key: str,
    dimensions: int | None = None,
) -> list[float] | None:
    """
    Embed a single text string.

    Parameters
    ----------
    text : str
        Text to embed
    model : str
        Embedding model name
    base_url : str
        OpenAI-compatible API base URL
    api_key : str
        API key for authentication
    dimensions : int | None
        Optional custom embedding dimension (only supported by some models)

    Returns
    -------
    list[float] | None
        Embedding vector, or None if the API call fails
    """
    vectors, _errors = embed_texts([text], model, base_url, api_key, dimensions)
    return vectors[0] if vectors else None
