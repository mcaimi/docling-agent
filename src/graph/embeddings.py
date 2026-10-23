#!/usr/bin/env python
"""OpenAI-compatible embedding client and batch embedding helpers."""

# import the openai client
try:
    from openai import OpenAI
    import httpx
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

import os

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

# define the embedding client cache
_embedding_client: OpenAI | None = None
_embedding_client_config: tuple[str, str] | None = None


# get the embedding client
# uses the connection params to configure the client
# connection settings are stored in the graph state
def get_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Return a cached OpenAI client with connection pooling, recreating it if credentials change."""
    global _embedding_client, _embedding_client_config
    current_config = (base_url, api_key)

    if _embedding_client is None or _embedding_client_config != current_config:
        # Configure httpx client with connection pooling
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        _embedding_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        _embedding_client_config = current_config

    return _embedding_client


def _embed_batch(
    client: OpenAI, texts: list[str], model: str
) -> list[list[float] | None]:
    """
    Embed a single batch of texts (up to _EMBEDDING_BATCH_SIZE items).

    Returns embeddings in the same order as input texts. On API error, all
    positions in this batch are set to None.
    """
    try:
        response = client.embeddings.create(
            input=texts,
            model=model,
        )
        # Sort by index to ensure correct order (API may return out of order)
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_embeddings]
    except Exception as e:
        print(f"Error generating embeddings for batch of {len(texts)} texts: {str(e)}")
        return [None] * len(texts)


# embed the texts
# uses the embedding client to generate embeddings for the texts
# returns a list of embeddings
def embed_texts(
    texts: list[str], model: str, base_url: str, api_key: str
) -> list[list[float] | None]:
    """
    Request embeddings for all texts using batched API calls.

    Splits input into batches of up to 2048 items (OpenAI API limit) and makes
    parallel-capable requests. On API errors for a batch, those positions are
    ``None`` and the error is printed.

    Returns embeddings in the same order as input texts.
    """
    if not texts:
        return []

    client = get_embedding_client(base_url, api_key)
    all_embeddings: list[list[float] | None] = []
    embedding_dimensions: int | None = None

    # Process in batches
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        batch_embeddings = _embed_batch(client, batch, model)
        all_embeddings.extend(batch_embeddings)

        # Capture dimensions from first successful embedding
        if embedding_dimensions is None:
            for emb in batch_embeddings:
                if emb is not None:
                    embedding_dimensions = len(emb)
                    break

    if enable_tracing:
        with _ml.start_span("embed_texts"):
            _ml.set_tag("model", model)
            _ml.set_tag("text_count", len(texts))
            _ml.set_tag("batch_count", (len(texts) + _EMBEDDING_BATCH_SIZE - 1) // _EMBEDDING_BATCH_SIZE)
            _ml.set_tag("batch_size", _EMBEDDING_BATCH_SIZE)
            if embedding_dimensions is not None:
                _ml.set_tag("embedding_dimensions", embedding_dimensions)
            _ml.set_tag("success_count", sum(1 for e in all_embeddings if e is not None))
            _ml.set_tag("failure_count", sum(1 for e in all_embeddings if e is None))

    return all_embeddings


def embed_single_text(text: str, model: str, base_url: str, api_key: str) -> list[float] | None:
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

    Returns
    -------
    list[float] | None
        Embedding vector, or None if the API call fails
    """
    result = embed_texts([text], model, base_url, api_key)
    return result[0] if result else None
