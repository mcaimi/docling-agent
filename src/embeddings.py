"""OpenAI-compatible embedding client and batch embedding helpers."""

import os

from openai import OpenAI

_embedding_client: OpenAI | None = None


def get_embedding_client() -> OpenAI:
    """Return a singleton OpenAI client configured from environment variables."""
    global _embedding_client
    if _embedding_client is None:
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set EMBEDDING_API_KEY (or OPENAI_API_KEY) for the embedding endpoint."
            )
        _embedding_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )
    return _embedding_client


def embed_texts(texts: list[str], model: str) -> list[list[float] | None]:
    """
    Request one embedding per string. On API errors for an item, that position is
    ``None`` and the error is printed.
    """
    if not texts:
        return []

    client = get_embedding_client()
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
