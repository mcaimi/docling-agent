#!/usr/bin/env python
"""LangChain vector store connectors for persisting precomputed chunk embeddings."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
except ImportError as e:
    raise ImportError(f"langchain-core and tenacity are required: {e}") from e

try:
    from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
    from ..utils.mlflow_tracing import run_context, start_child_span, end_span
except ImportError as e:
    raise ImportError(f"Failed to import circuit_breaker: {e}") from e

# check if tracing is enabled
enable_tracing = int(os.getenv("ENABLE_TRACING", "0")) == 1

if enable_tracing:
    try:
        import mlflow as _ml
    except ImportError:
        print("MLflow is not installed. Tracing is disabled.")
        enable_tracing = False

# supported vector store backends
# none: no vector store, skip the vector store step e.g. keep the embeddings in memory
# chroma: use chroma as the vector store
# milvus: use milvus as the vector store
VectorStoreBackend = Literal["none", "chroma", "milvus"]

# supported write modes
WriteMode = Literal["insert", "upsert", "skip_existing"]

# global vector store cache for connection reuse. Guarded by a lock: pipeline
# tasks run concurrently (one per document in a batch), and without it two
# threads could race through the check-then-create-then-store sequence below
# and each build their own backend connection for the same cache key, silently
# discarding one (a real resource leak for backends like Chroma that hold a
# persistent-directory file lock, or Milvus's connection pool).
_vector_store_cache: dict[tuple, VectorStore] = {}
_vector_store_cache_config: dict[tuple, dict[str, Any]] = {}
_vector_store_cache_lock = threading.Lock()

# global circuit breakers per backend, same race as above without a lock.
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.Lock()


class _NoOpEmbeddings(Embeddings):
    """Placeholder; vectors are supplied via add_embeddings at insert time."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("embeddings are precomputed by the pipeline")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("embeddings are precomputed by the pipeline")


class QueryEmbeddings(Embeddings):
    """Embeddings class for query-time vector generation using OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str, api_key: str, dimensions: int | None = None):
        """
        Initialize query embeddings.

        Parameters
        ----------
        model : str
            Embedding model name
        base_url : str
            OpenAI-compatible API base URL
        api_key : str
            API key for authentication
        dimensions : int | None
            Must match whatever dimension was used to embed the stored chunks,
            or similarity search against the vector store will fail/be meaningless.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        try:
            from .embeddings import embed_texts
        except ImportError as e:
            raise ImportError(f"embeddings module required: {e}") from e

        vectors, errors = embed_texts(texts, self.model, self.base_url, self.api_key, self.dimensions)
        for error in errors:
            print(f"Query embedding failed: {error}")
        # Filter out None values and return only valid embeddings
        return [v for v in vectors if v is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        vectors = self.embed_documents([text])
        if not vectors:
            raise ValueError("Failed to generate embedding for query")
        return vectors[0]


# build cache key for vector store instances
def _build_cache_key(config: dict[str, Any]) -> tuple:
    """Build hashable cache key from config."""
    backend = _backend_from_config(config)
    if backend == "none":
        return ("none",)

    collection = _collection_name(config)

    if backend == "chroma":
        return (
            "chroma",
            collection,
            config.get("chroma_host"),
            config.get("chroma_port", 8000),
            config.get("chroma_persist_directory"),
        )
    elif backend == "milvus":
        return (
            "milvus",
            collection,
            config.get("milvus_uri", "http://localhost:19530"),
            config.get("milvus_token"),
            config.get("milvus_db_name"),
        )
    return (backend, collection)


# get or create circuit breaker for a backend
def _get_circuit_breaker(backend: str, config: dict[str, Any]) -> CircuitBreaker:
    """Get or create a circuit breaker for the specified backend."""
    circuit_breaker = _circuit_breakers.get(backend)
    if circuit_breaker is not None:
        return circuit_breaker

    with _circuit_breakers_lock:
        circuit_breaker = _circuit_breakers.get(backend)
        if circuit_breaker is not None:
            return circuit_breaker

        threshold = config.get("vector_store_circuit_breaker_threshold", 5)
        timeout = config.get("vector_store_circuit_breaker_timeout", 60)
        circuit_breaker = CircuitBreaker(
            failure_threshold=threshold,
            timeout=float(timeout),
            name=f"vector_store_{backend}",
        )
        _circuit_breakers[backend] = circuit_breaker
        return circuit_breaker


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


# create a new vector store instance
def _create_vector_store(config: dict[str, Any]) -> VectorStore:
    """Create a new vector store instance (internal use only)."""
    backend = _backend_from_config(config)
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

            # Add Chroma-specific configurations
            collection_metadata = config.get("chroma_collection_metadata", {})
            if collection_metadata:
                kwargs["collection_metadata"] = collection_metadata

            return Chroma(**kwargs)

        case "milvus":
            try:
                from langchain_milvus import Milvus
            except ImportError as e:
                raise ImportError(
                    "Milvus backend requires optional deps: uv sync --extra milvus"
                ) from e
            connection_args: dict[str, Any] = {
                "uri": config.get("milvus_uri", "http://localhost:19530")
            }
            token = config.get("milvus_token")
            if token:
                connection_args["token"] = token
            db_name = config.get("milvus_db_name")
            if db_name:
                connection_args["db_name"] = db_name

            # Add consistency level if specified
            consistency_level = config.get("milvus_consistency_level")
            if consistency_level:
                connection_args["consistency_level"] = consistency_level

            partition_name = config.get("milvus_partition_name")

            kwargs: dict[str, Any] = {
                "embedding_function": noop,
                "collection_name": collection,
                "connection_args": connection_args,
                "auto_id": False,
            }

            if partition_name:
                kwargs["partition_key_field"] = partition_name

            return Milvus(**kwargs)

        case _:
            raise ValueError(f"Unsupported vector_store_backend: {backend}")


# get or create cached vector store instance
def get_or_create_vector_store(config: dict[str, Any]) -> VectorStore | None:
    """
    Return a cached LangChain vector store for the configured backend, or None if disabled.

    Uses singleton pattern to reuse connections across calls with same configuration.
    """
    backend = _backend_from_config(config)
    if backend == "none":
        return None

    cache_key = _build_cache_key(config)

    # Check if we have a cached instance with matching config
    if _vector_store_cache_config.get(cache_key) == config:
        cached = _vector_store_cache.get(cache_key)
        if cached is not None:
            return cached

    with _vector_store_cache_lock:
        # re-check inside the lock: another thread may have just built this
        if _vector_store_cache_config.get(cache_key) == config:
            cached = _vector_store_cache.get(cache_key)
            if cached is not None:
                return cached

        # Create new instance and cache it
        vector_store = _create_vector_store(config)
        _vector_store_cache[cache_key] = vector_store
        _vector_store_cache_config[cache_key] = config.copy()

        return vector_store


# backward compatibility alias
def get_vector_store(config: dict[str, Any]) -> VectorStore | None:
    """
    Return a LangChain vector store for the configured backend, or None if disabled.

    This is an alias for get_or_create_vector_store() for backward compatibility.
    """
    return get_or_create_vector_store(config)


# check if vector exists (for upsert logic)
def _check_ids_exist(vector_store: VectorStore, ids: list[str]) -> set[str]:
    """
    Check which IDs already exist in the vector store.

    Returns a set of existing IDs. Falls back to empty set if backend doesn't support get.
    """
    if not ids:
        return set()

    try:
        # Try to get existing documents
        if hasattr(vector_store, "get"):
            result = vector_store.get(ids=ids)
            if result and hasattr(result, "get"):
                return set(result.get("ids", []))
    except Exception:
        # If get is not supported or fails, assume no existing IDs
        pass

    return set()


# delete IDs from vector store
def _delete_ids(vector_store: VectorStore, ids: list[str]) -> None:
    """Delete documents by IDs from the vector store."""
    if not ids:
        return

    if hasattr(vector_store, "delete"):
        vector_store.delete(ids=ids)
    else:
        raise NotImplementedError(
            f"Backend {type(vector_store).__name__} does not support delete operation"
        )


# insert a batch of embeddings into the vector store (with retry)
def _insert_batch_with_retry(
    vector_store: VectorStore,
    texts: list[str],
    vectors: list[list[float]],
    metadatas: list[dict[str, str | int | float | bool]],
    ids: list[str],
    *,
    batch_size: int,
    retry_attempts: int = 3,
    retry_backoff: int = 1,
) -> None:
    """
    Insert one batch using the backend-specific LangChain connector API with retry logic.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    texts : list[str]
        Text content
    vectors : list[list[float]]
        Embedding vectors
    metadatas : list[dict]
        Metadata for each chunk
    ids : list[str]
        Chunk IDs
    batch_size : int
        Batch size for the operation
    retry_attempts : int
        Number of retry attempts for transient failures
    retry_backoff : int
        Backoff multiplier for exponential backoff
    """

    @retry(
        stop=stop_after_attempt(retry_attempts),
        wait=wait_exponential(multiplier=retry_backoff, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _do_insert():
        if hasattr(vector_store, "add_embeddings"):
            vector_store.add_embeddings(
                texts=texts,
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids,
                batch_size=batch_size,
            )
        else:
            vector_store.add_texts(
                texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=vectors,
            )

    _do_insert()


# insert a batch (backward compatibility)
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
    _insert_batch_with_retry(
        vector_store,
        texts,
        vectors,
        metadatas,
        ids,
        batch_size=batch_size,
        retry_attempts=3,
        retry_backoff=1,
    )


# store the chunk embeddings in the vector store
def store_chunk_embeddings(
    vector_store: VectorStore,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float] | None],
    *,
    batch_size: int = 200,
    write_mode: WriteMode = "insert",
    max_workers: int = 4,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    parent_span: Any = None,
) -> dict[str, Any]:
    """
    Persist precomputed embeddings for chunk records.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    chunks : list[dict]
        Chunk records with content and metadata
    embeddings : list[list[float] | None]
        Precomputed embeddings (None values are skipped)
    batch_size : int
        Number of vectors per insert batch (default: 200)
    write_mode : WriteMode
        Write behavior: "insert" (default), "upsert", or "skip_existing"
    max_workers : int
        Number of parallel batch workers (default: 4)
    config : dict | None
        Configuration dict for circuit breaker and retry settings
    run_id : str | None
        When set, records a child span of ``parent_span`` on that MLflow run
        for this call (see ``utils.mlflow_tracing``)
    parent_span : Any
        Span this call's span should nest under (typically store_doc's span)

    Returns
    -------
    dict
        Statistics: stored_count, skipped_count, ids, upserted_count, deleted_count,
        retry_count, failed_chunks
    """
    import time

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    if config is None:
        config = {}

    start_time = time.time()
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
        return {
            "stored_count": 0,
            "skipped_count": skipped_count,
            "ids": [],
            "upserted_count": 0,
            "deleted_count": 0,
            "retry_count": 0,
            "failed_chunks": [],
        }

    # Handle upsert mode: check for existing IDs and delete them first
    deleted_count = 0
    if write_mode == "upsert":
        existing_ids = _check_ids_exist(vector_store, ids)
        if existing_ids:
            try:
                _delete_ids(vector_store, list(existing_ids))
                deleted_count = len(existing_ids)
            except Exception as e:
                print(f"Warning: Failed to delete existing IDs for upsert: {e}")

    elif write_mode == "skip_existing":
        existing_ids = _check_ids_exist(vector_store, ids)
        if existing_ids:
            # Filter out existing IDs
            filtered_data = [
                (t, v, m, i)
                for t, v, m, i in zip(texts, vectors, metadatas, ids)
                if i not in existing_ids
            ]
            if filtered_data:
                texts, vectors, metadatas, ids = zip(*filtered_data)
                texts, vectors, metadatas, ids = (
                    list(texts),
                    list(vectors),
                    list(metadatas),
                    list(ids),
                )
            else:
                # All IDs already exist
                return {
                    "stored_count": 0,
                    "skipped_count": skipped_count + len(existing_ids),
                    "ids": [],
                    "upserted_count": 0,
                    "deleted_count": 0,
                    "retry_count": 0,
                    "failed_chunks": [],
                }

    # Get retry configuration
    retry_attempts = config.get("vector_store_retry_attempts", 3)
    retry_backoff = config.get("vector_store_retry_backoff_multiplier", 1)

    # Get circuit breaker
    backend = type(vector_store).__name__.lower()
    circuit_breaker = _get_circuit_breaker(backend, config)

    # Batch operations with parallel execution
    stored_ids: list[str] = []
    failed_chunks: list[dict[str, Any]] = []
    total_retry_count = 0

    # Create batches
    batches = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batches.append(
            {
                "texts": texts[start:end],
                "vectors": vectors[start:end],
                "metadatas": metadatas[start:end],
                "ids": ids[start:end],
            }
        )

    def process_batch(batch_data: dict) -> dict[str, Any]:
        """Process a single batch with circuit breaker and retry."""
        batch_result = {
            "ids": batch_data["ids"],
            "success": False,
            "retry_count": 0,
            "error": None,
        }

        try:
            # Wrap in circuit breaker
            circuit_breaker.call(
                _insert_batch_with_retry,
                vector_store,
                batch_data["texts"],
                batch_data["vectors"],
                batch_data["metadatas"],
                batch_data["ids"],
                batch_size=batch_size,
                retry_attempts=retry_attempts,
                retry_backoff=retry_backoff,
            )
            batch_result["success"] = True
        except CircuitBreakerError as e:
            batch_result["error"] = f"Circuit breaker open: {e}"
        except Exception as e:
            batch_result["error"] = str(e)

        return batch_result

    # Execute batches in parallel
    if max_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    stored_ids.extend(result["ids"])
                else:
                    failed_chunks.extend(
                        [{"id": id_, "error": result["error"]} for id_ in result["ids"]]
                    )
                total_retry_count += result["retry_count"]
    else:
        # Sequential processing
        for batch in batches:
            result = process_batch(batch)
            if result["success"]:
                stored_ids.extend(result["ids"])
            else:
                failed_chunks.extend(
                    [{"id": id_, "error": result["error"]} for id_ in result["ids"]]
                )
            total_retry_count += result["retry_count"]

    duration_ms = (time.time() - start_time) * 1000

    # MLflow tracing. Re-enters mlflow_run_id (when present in config) as the
    # active run before logging -- this is typically called from a LangGraph
    # task worker thread, whose context doesn't inherit the active run, so
    # without it MLflow would spawn a new orphan run to hold these metrics.
    if enable_tracing:
        with run_context(config.get("mlflow_run_id")):
            _ml.log_metric("stored_count", len(stored_ids))
            _ml.log_metric("skipped_count", skipped_count)
            _ml.log_metric("upserted_count", deleted_count if write_mode == "upsert" else 0)
            _ml.log_metric("deleted_count", deleted_count)
            _ml.log_metric("retry_count", total_retry_count)
            _ml.log_metric("failed_count", len(failed_chunks))
            _ml.log_metric("duration_ms", duration_ms)
            if len(stored_ids) > 0:
                _ml.log_metric(
                    "throughput_vectors_per_sec", len(stored_ids) / (duration_ms / 1000)
                )

    # When called from store_doc (run_id/parent_span passed explicitly), also
    # record this as a child span of that document's trace.
    span = start_child_span(run_id, parent_span, "store_chunk_embeddings")
    end_span(
        span,
        write_mode=write_mode,
        batch_size=batch_size,
        max_workers=max_workers,
        stored_count=len(stored_ids),
        skipped_count=skipped_count,
        retry_count=total_retry_count,
        failed_count=len(failed_chunks),
        duration_ms=duration_ms,
    )

    return {
        "stored_count": len(stored_ids),
        "skipped_count": skipped_count,
        "ids": stored_ids,
        "upserted_count": deleted_count if write_mode == "upsert" else 0,
        "deleted_count": deleted_count,
        "retry_count": total_retry_count,
        "failed_chunks": failed_chunks,
        "duration_ms": duration_ms,
    }


# delete chunk embeddings from the vector store
def delete_chunk_embeddings(
    vector_store: VectorStore,
    *,
    ids: list[str] | None = None,
    file_paths: list[str] | None = None,
    metadata_filters: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Delete chunk embeddings from the vector store.

    Supports deletion by chunk IDs, file paths, or metadata filters.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    ids : list[str] | None
        Specific chunk IDs to delete
    file_paths : list[str] | None
        Delete all chunks from these file paths
    metadata_filters : dict | None
        Delete chunks matching these metadata filters
    config : dict | None
        Configuration dict for circuit breaker settings

    Returns
    -------
    dict
        Statistics: deleted_count, failed_count, errors
    """
    import time

    if config is None:
        config = {}

    start_time = time.time()
    ids_to_delete: list[str] = []

    # Collect IDs based on different delete criteria
    if ids:
        ids_to_delete.extend(ids)

    if file_paths:
        # Generate all possible chunk IDs for each file path
        # This is a heuristic; ideally we'd query the vector store
        for file_path in file_paths:
            # Try to get chunks from this file path
            # Note: This requires the backend to support metadata filtering
            if hasattr(vector_store, "get"):
                try:
                    # Try to get all documents from this file
                    results = vector_store.get(
                        where={"file_path": file_path} if metadata_filters is None else None
                    )
                    if results and "ids" in results:
                        ids_to_delete.extend(results["ids"])
                except Exception as e:
                    print(f"Warning: Could not retrieve IDs for file {file_path}: {e}")

    if metadata_filters and not file_paths:
        # Query by metadata filters
        if hasattr(vector_store, "get"):
            try:
                results = vector_store.get(where=metadata_filters)
                if results and "ids" in results:
                    ids_to_delete.extend(results["ids"])
            except Exception as e:
                print(f"Warning: Could not retrieve IDs with metadata filters: {e}")

    if not ids_to_delete:
        return {
            "deleted_count": 0,
            "failed_count": 0,
            "errors": [],
            "duration_ms": 0,
        }

    # Remove duplicates
    ids_to_delete = list(set(ids_to_delete))

    # Get circuit breaker
    backend = type(vector_store).__name__.lower()
    circuit_breaker = _get_circuit_breaker(backend, config)

    deleted_count = 0
    errors = []

    try:
        circuit_breaker.call(_delete_ids, vector_store, ids_to_delete)
        deleted_count = len(ids_to_delete)
    except CircuitBreakerError as e:
        errors.append(f"Circuit breaker open: {e}")
    except Exception as e:
        errors.append(f"Delete failed: {e}")

    duration_ms = (time.time() - start_time) * 1000

    # MLflow tracing
    if enable_tracing:
        with run_context(config.get("mlflow_run_id")), _ml.start_span("delete_chunk_embeddings"):
            _ml.log_metric("deleted_count", deleted_count)
            _ml.log_metric("failed_count", len(ids_to_delete) - deleted_count)
            _ml.log_metric("duration_ms", duration_ms)

    return {
        "deleted_count": deleted_count,
        "failed_count": len(ids_to_delete) - deleted_count,
        "errors": errors,
        "duration_ms": duration_ms,
    }


# query the vector store
def query_vector_store(
    vector_store: VectorStore,
    query: str,
    *,
    k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
    embedding_config: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Query the vector store for similar chunks.

    Parameters
    ----------
    vector_store : VectorStore
        The vector store instance
    query : str
        Query text
    k : int
        Number of results to return (default: 5)
    metadata_filter : dict | None
        Filter results by metadata
    embedding_config : dict
        Configuration for embedding generation (model, base_url, api_key)
    config : dict | None
        Configuration dict for circuit breaker settings

    Returns
    -------
    list[dict]
        Retrieved chunks with content, metadata, and similarity scores
    """
    import time

    if config is None:
        config = {}

    start_time = time.time()

    # Create query embeddings instance
    query_embedder = QueryEmbeddings(
        model=embedding_config.get("embedding_model", "text-embedding-ada-002"),
        base_url=embedding_config.get("embedding_base_url", "https://api.openai.com/v1"),
        api_key=embedding_config.get("embedding_api_key", ""),
        dimensions=embedding_config.get("embedding_dimensions"),
    )

    # Get circuit breaker
    backend = type(vector_store).__name__.lower()
    circuit_breaker = _get_circuit_breaker(backend, config)

    results = []

    try:
        # Generate query embedding
        query_vector = query_embedder.embed_query(query)

        # Perform similarity search
        def _do_search():
            if metadata_filter:
                return vector_store.similarity_search_by_vector(
                    query_vector, k=k, filter=metadata_filter
                )
            return vector_store.similarity_search_by_vector(query_vector, k=k)

        docs = circuit_breaker.call(_do_search)

        # Format results
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None),
                }
            )

    except CircuitBreakerError as e:
        print(f"Query failed: Circuit breaker open: {e}")
    except Exception as e:
        print(f"Query failed: {e}")

    duration_ms = (time.time() - start_time) * 1000

    # MLflow tracing
    if enable_tracing:
        with run_context(config.get("mlflow_run_id")), _ml.start_span("query_vector_store"):
            _ml.set_tag("k", k)
            _ml.set_tag("has_filter", metadata_filter is not None)
            _ml.log_metric("result_count", len(results))
            _ml.log_metric("duration_ms", duration_ms)

    return results
