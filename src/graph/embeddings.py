#!/usr/bin/env python
"""OpenAI-compatible embedding client and batch embedding helpers."""

# import the openai client
try:
    from openai import OpenAI
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

# define the embedding client (singleton)
_embedding_client: OpenAI = None

# get the embedding client
# uses the connection params to configure the client
# connection settings are stored in the graph state
def get_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Return a singleton OpenAI client configured from agent settings (.env + config defaults)."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return _embedding_client


# embed the texts
# uses the embedding client to generate embeddings for the texts
# returns a list of embeddings
def embed_texts(texts: list[str], model: str, base_url: str, api_key: str) -> list[list[float] | None]:
    """
    Request one embedding per string. On API errors for an item, that position is
    ``None`` and the error is printed.
    """
    if not texts:
        return []

    client = get_embedding_client(base_url, api_key)
    embeddings: list[list[float] | None] = []
    embedding_dimensions: int | None = None

    for content in texts:
        try:
            response = client.embeddings.create(
                input=content,
                model=model,
            )
            _vec = response.data[0].embedding
            embeddings.append(_vec)
            if embedding_dimensions is None and _vec:
                _dims = len(_vec)
        except Exception as e:
            print(f"Error generating embedding for content: {str(e)}")
            embeddings.append(None)

    if enable_tracing:
        with _ml.start_span("embed_texts"):
            _ml.set_tag("model", model)
            _ml.set_tag("text_count", len(texts))
            if embedding_dimensions is not None:
                _ml.set_tag("embedding_dimensions", _dims)
            _ml.set_tag("success_count", sum(1 for e in embeddings if e is not None))
            _ml.set_tag("failure_count", sum(1 for e in embeddings if e is None))

    return embeddings
