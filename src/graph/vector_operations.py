#!/usr/bin/env python
"""Higher-level vector store operations for document lifecycle management."""

from typing import Any

try:
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(f"langchain-core is required: {e}") from e

try:
    from .vector_store import delete_chunk_embeddings, _chunk_id
except ImportError as e:
    raise ImportError(f"Failed to import vector_store: {e}") from e


def delete_document(
    vector_store: VectorStore,
    file_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Delete all chunks from a specific document.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    file_path : str
        Path to the document whose chunks should be deleted
    config : dict | None
        Configuration dict for circuit breaker settings

    Returns
    -------
    dict
        Statistics: deleted_count, failed_count, errors
    """
    return delete_chunk_embeddings(
        vector_store, file_paths=[file_path], config=config
    )


def delete_documents_batch(
    vector_store: VectorStore,
    file_paths: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Delete all chunks from multiple documents.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    file_paths : list[str]
        Paths to documents whose chunks should be deleted
    config : dict | None
        Configuration dict for circuit breaker settings

    Returns
    -------
    dict
        Statistics: deleted_count, failed_count, errors
    """
    return delete_chunk_embeddings(
        vector_store, file_paths=file_paths, config=config
    )


def replace_document_chunks(
    vector_store: VectorStore,
    file_path: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float] | None],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Replace all chunks for a document (delete old, insert new).

    This is useful for document versioning: delete the old version's chunks,
    then insert the new version.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    file_path : str
        Path to the document being replaced
    chunks : list[dict]
        New chunk records with content and metadata
    embeddings : list[list[float] | None]
        Precomputed embeddings for new chunks
    config : dict | None
        Configuration dict for vector store settings

    Returns
    -------
    dict
        Combined statistics from delete and insert operations
    """
    from .vector_store import store_chunk_embeddings

    if config is None:
        config = {}

    # Delete old chunks
    delete_result = delete_document(vector_store, file_path, config)

    # Insert new chunks (use insert mode, not upsert, since we already deleted)
    store_result = store_chunk_embeddings(
        vector_store,
        chunks,
        embeddings,
        batch_size=config.get("vector_store_batch_size", 200),
        write_mode="insert",
        max_workers=config.get("vector_store_max_batch_workers", 4),
        config=config,
    )

    return {
        "deleted_count": delete_result["deleted_count"],
        "delete_errors": delete_result.get("errors", []),
        "stored_count": store_result["stored_count"],
        "skipped_count": store_result["skipped_count"],
        "retry_count": store_result.get("retry_count", 0),
        "failed_chunks": store_result.get("failed_chunks", []),
    }


def cleanup_old_versions(
    vector_store: VectorStore,
    file_paths: list[str],
    keep_latest: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Clean up old document versions from the vector store.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    file_paths : list[str]
        Paths to documents to clean up
    keep_latest : bool
        If True, keep the latest version (based on chunk metadata timestamps)
    config : dict | None
        Configuration dict for circuit breaker settings

    Returns
    -------
    dict
        Statistics: deleted_count, failed_count, errors

    Notes
    -----
    This function requires chunk metadata to include version or timestamp information.
    If keep_latest is False, all versions are deleted.
    """
    if not keep_latest:
        return delete_documents_batch(vector_store, file_paths, config)

    # TODO: Implement version-aware cleanup
    # This would require:
    # 1. Query all chunks for each file_path
    # 2. Group by version/timestamp
    # 3. Identify old versions
    # 4. Delete old version IDs
    # For now, this is a placeholder
    raise NotImplementedError(
        "Version-aware cleanup requires metadata with version/timestamp fields"
    )
