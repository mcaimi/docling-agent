#!/usr/bin/env python
"""LangChain vector store connectors for persisting precomputed chunk embeddings."""

from typing import Any, Literal

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(f"langchain-core is required: {e}") from e

# supported vector store backends
# none: no vector store, skip the vector store step e.g. keep the embeddings in memory
# chroma: use chroma as the vector store
# milvus: use milvus as the vector store
VectorStoreBackend = Literal["none", "chroma", "milvus"]


class _NoOpEmbeddings(Embeddings):
    """Placeholder; vectors are supplied via add_embeddings at insert time."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("embeddings are precomputed by the pipeline")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("embeddings are precomputed by the pipeline")


# get the vector store backend from the configuration
def _backend_from_config(config: dict[str, Any]) -> VectorStoreBackend:
    raw = (config.get("vector_store_backend") or "none").strip().lower()
    if raw in ("none", ""):
        return "none"
    if raw in ("chroma", "milvus"):
        return raw  # type: ignore[return-value]
    raise ValueError(
        f"vector_store_backend must be 'none', 'chroma', or 'milvus', got {raw!r}"
    )


# get the collection name from the configuration
def _collection_name(config: dict[str, Any]) -> str:
    return config.get("vector_store_collection") or "docling_chunks"


# sanitize the metadata to only include scalar values supported by Chroma/Milvus
def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only scalar metadata values supported by Chroma/Milvus."""
    out: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif value is not None:
            out[key] = str(value)
    return out


# generate a unique id for a chunk
def _chunk_id(metadata: dict[str, Any]) -> str:
    file_path = metadata.get("file_path", "unknown")
    chunk_index = metadata.get("chunk_index", 0)
    return f"{file_path}#chunk_{chunk_index}"


# get the vector store for the configured backend
def get_vector_store(config: dict[str, Any]) -> VectorStore | None:
    """Return a LangChain vector store for the configured backend, or None if disabled."""
    backend = _backend_from_config(config)
    if backend == "none":
        return None

    collection = _collection_name(config)
    noop = _NoOpEmbeddings()

    # branch by selected backend
    match backend:
        case "chroma":
            try:
                from langchain_chroma import Chroma
            except ImportError as e:
                raise ImportError(
                    "Chroma backend requires optional deps: uv sync --extra chroma"
                ) from e
            kwargs: dict[str, Any] = {
                "collection_name": collection,
                "embedding_function": None,
            }
            persist_dir = config.get("chroma_persist_directory")
            if persist_dir:
                kwargs["persist_directory"] = persist_dir
            host = config.get("chroma_host")
            if host:
                kwargs["host"] = host
                kwargs["port"] = config.get("chroma_port", 8000)
            return Chroma(**kwargs)
        case "milvus":
            try:
                from langchain_milvus import Milvus
            except ImportError as e:
                raise ImportError(
                    "Milvus backend requires optional deps: uv sync --extra milvus"
                ) from e
            connection_args: dict[str, Any] = {"uri": config.get("milvus_uri", "http://localhost:19530")}
            token = config.get("milvus_token")
            if token:
                connection_args["token"] = token
            db_name = config.get("milvus_db_name")
            if db_name:
                connection_args["db_name"] = db_name
            return Milvus(
                embedding_function=noop,
                collection_name=collection,
                connection_args=connection_args,
                auto_id=False,
            )
        case _:
            raise ValueError(f"Unsupported vector_store_backend: {backend}")


# insert a batch of embeddings into the vector store
def _insert_batch(
    vector_store: VectorStore,
    texts: list[str],
    vectors: list[list[float]],
    metadatas: list[dict[str, str | int | float | bool]],
    ids: list[str],
    *,
    batch_size: int,
) -> None:
    """Insert one batch using the backend-specific LangChain connector API."""
    if hasattr(vector_store, "add_embeddings"):
        vector_store.add_embeddings(
            texts=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
        )
        return

    vector_store.add_texts(
        texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=vectors,
    )


# store the chunk embeddings in the vector store
def store_chunk_embeddings(
    vector_store: VectorStore,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float] | None],
    *,
    batch_size: int = 200,
) -> dict[str, Any]:
    """
    Persist precomputed embeddings for chunk records.

    Skips pairs where embedding is None. Inserts in sub-batches of ``batch_size``.
    Re-running with the same chunk IDs may overwrite or fail depending on the backend.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    texts: list[str] = []
    vectors: list[list[float]] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    ids: list[str] = []
    skipped_count = 0

    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            skipped_count += 1
            continue
        meta = _sanitize_metadata(chunk.get("metadata", {}))
        texts.append(chunk["content"])
        vectors.append(embedding)
        metadatas.append(meta)
        ids.append(_chunk_id(meta))

    if not texts:
        return {"stored_count": 0, "skipped_count": skipped_count, "ids": []}

    stored_ids: list[str] = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_vectors = vectors[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]
        _insert_batch(
            vector_store,
            batch_texts,
            batch_vectors,
            batch_metas,
            batch_ids,
            batch_size=batch_size,
        )
        stored_ids.extend(batch_ids)

    return {
        "stored_count": len(stored_ids),
        "skipped_count": skipped_count,
        "ids": stored_ids,
    }
