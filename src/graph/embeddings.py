#!/usr/bin/env python
"""OpenAI-compatible embedding client and batch embedding helpers."""

# import the openai client
try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}") from e

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

    for content in texts:
        try:
            response = client.embeddings.create(
                input=content,
                model=model,
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for content: {str(e)}")
            embeddings.append(None)

    return embeddings
