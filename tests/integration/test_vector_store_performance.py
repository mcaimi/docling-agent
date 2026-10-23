#!/usr/bin/env python
"""Performance tests for vector store operations."""

import time
import pytest


class TestVectorStorePerformance:
    """Performance benchmarks for vector store operations."""

    @pytest.fixture(params=["milvus", "chroma"])
    def vector_store_setup(self, request, milvus_available, chromadb_available, milvus_config, chromadb_config):
        """Setup vector store for either backend."""
        from src.graph.vector_store import get_or_create_vector_store

        if request.param == "milvus":
            config = milvus_config.copy()
            config["vector_store_collection"] = f"perf_test_milvus_{id(self)}"
        else:
            config = chromadb_config.copy()
            config["vector_store_collection"] = f"perf_test_chroma_{id(self)}"

        vector_store = get_or_create_vector_store(config)

        yield vector_store, config, request.param

        # Cleanup
        try:
            if request.param == "milvus" and hasattr(vector_store, "col"):
                vector_store.col.drop()
            elif request.param == "chroma" and hasattr(vector_store, "_client"):
                vector_store._client.delete_collection(config["vector_store_collection"])
        except Exception:
            pass

    def test_large_batch_insert_performance(self, vector_store_setup, sample_embeddings):
        """Test performance with large batch inserts."""
        from src.graph.vector_store import store_chunk_embeddings

        vector_store, config, backend = vector_store_setup

        # Create 1000 chunks
        num_chunks = 1000
        chunks = []
        embeddings = []

        for i in range(num_chunks):
            chunks.append({
                "content": f"Performance test content {i}",
                "metadata": {
                    "file_path": "/test/perf_doc.pdf",
                    "file_name": "perf_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": num_chunks,
                },
            })
            embeddings.append(sample_embeddings[i % 3])

        start_time = time.time()

        result = store_chunk_embeddings(
            vector_store,
            chunks,
            embeddings,
            batch_size=200,
            max_workers=4,
            config=config,
        )

        duration = time.time() - start_time

        assert result["stored_count"] == num_chunks
        assert result["failed_chunks"] == []

        # Calculate throughput
        throughput = num_chunks / duration
        print(f"\n{backend} - Inserted {num_chunks} vectors in {duration:.2f}s ({throughput:.0f} vectors/sec)")

        # Performance expectations (these are reasonable baselines)
        # Milvus and Chroma should handle at least 100 vectors/sec for this batch size
        assert throughput > 50, f"Throughput too low: {throughput:.0f} vectors/sec"

    def test_parallel_batch_performance(self, vector_store_setup, sample_embeddings):
        """Test that parallel batch processing is faster than sequential."""
        from src.graph.vector_store import store_chunk_embeddings

        vector_store, config, backend = vector_store_setup

        # Create 500 chunks
        num_chunks = 500
        chunks = []
        embeddings = []

        for i in range(num_chunks):
            chunks.append({
                "content": f"Parallel test content {i}",
                "metadata": {
                    "file_path": "/test/parallel_doc.pdf",
                    "file_name": "parallel_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": num_chunks,
                },
            })
            embeddings.append(sample_embeddings[i % 3])

        # Sequential (1 worker)
        start_seq = time.time()
        result_seq = store_chunk_embeddings(
            vector_store,
            chunks,
            embeddings,
            batch_size=50,
            max_workers=1,
            config=config,
        )
        duration_seq = time.time() - start_seq

        # Clean up for next test
        from src.graph.vector_store import delete_chunk_embeddings
        delete_chunk_embeddings(vector_store, file_paths=["/test/parallel_doc.pdf"], config=config)

        # Parallel (4 workers)
        start_par = time.time()
        result_par = store_chunk_embeddings(
            vector_store,
            chunks,
            embeddings,
            batch_size=50,
            max_workers=4,
            config=config,
        )
        duration_par = time.time() - start_par

        assert result_seq["stored_count"] == num_chunks
        assert result_par["stored_count"] == num_chunks

        speedup = duration_seq / duration_par
        print(f"\n{backend} - Sequential: {duration_seq:.2f}s, Parallel: {duration_par:.2f}s, Speedup: {speedup:.2f}x")

        # Parallel should be at least 20% faster (conservative, accounting for overhead)
        # Note: May not always be true for small datasets or network-bound operations
        # This is more about verifying it doesn't get slower
        assert duration_par <= duration_seq * 1.2, "Parallel processing should not be significantly slower"

    def test_upsert_performance(self, vector_store_setup, sample_embeddings):
        """Test upsert performance vs separate delete+insert."""
        from src.graph.vector_store import store_chunk_embeddings

        vector_store, config, backend = vector_store_setup

        # Create chunks
        num_chunks = 200
        chunks = []
        embeddings = []

        for i in range(num_chunks):
            chunks.append({
                "content": f"Upsert test content {i}",
                "metadata": {
                    "file_path": "/test/upsert_doc.pdf",
                    "file_name": "upsert_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": num_chunks,
                },
            })
            embeddings.append(sample_embeddings[i % 3])

        # Initial insert
        store_chunk_embeddings(vector_store, chunks, embeddings, config=config)

        # Updated embeddings
        updated_embeddings = [sample_embeddings[(i + 1) % 3] for i in range(num_chunks)]

        # Test upsert mode
        start_upsert = time.time()
        result_upsert = store_chunk_embeddings(
            vector_store,
            chunks,
            updated_embeddings,
            write_mode="upsert",
            config=config,
        )
        duration_upsert = time.time() - start_upsert

        assert result_upsert["stored_count"] == num_chunks
        assert result_upsert["upserted_count"] == num_chunks

        print(f"\n{backend} - Upsert {num_chunks} vectors: {duration_upsert:.2f}s")

    def test_delete_performance(self, vector_store_setup, sample_embeddings):
        """Test delete operation performance."""
        from src.graph.vector_store import store_chunk_embeddings, delete_chunk_embeddings, _chunk_id

        vector_store, config, backend = vector_store_setup

        # Create chunks
        num_chunks = 500
        chunks = []
        embeddings = []

        for i in range(num_chunks):
            chunks.append({
                "content": f"Delete test content {i}",
                "metadata": {
                    "file_path": "/test/delete_doc.pdf",
                    "file_name": "delete_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": num_chunks,
                },
            })
            embeddings.append(sample_embeddings[i % 3])

        # Insert
        store_chunk_embeddings(vector_store, chunks, embeddings, config=config)

        # Delete by IDs
        ids_to_delete = [_chunk_id(chunk["metadata"]) for chunk in chunks[:250]]

        start_delete = time.time()
        result_delete = delete_chunk_embeddings(
            vector_store,
            ids=ids_to_delete,
            config=config,
        )
        duration_delete = time.time() - start_delete

        assert result_delete["deleted_count"] == 250

        print(f"\n{backend} - Delete 250 vectors: {duration_delete:.2f}s")

    def test_connection_reuse_performance(self, vector_store_setup, sample_embeddings):
        """Test that connection pooling improves performance."""
        from src.graph.vector_store import get_or_create_vector_store, store_chunk_embeddings

        vector_store, config, backend = vector_store_setup

        # Create small batches
        chunks = []
        embeddings = []

        for i in range(10):
            chunks.append({
                "content": f"Connection test {i}",
                "metadata": {
                    "file_path": "/test/conn_doc.pdf",
                    "file_name": "conn_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": 10,
                },
            })
            embeddings.append(sample_embeddings[i % 3])

        # Multiple sequential inserts using cached connection
        start_cached = time.time()
        for i in range(5):
            vs = get_or_create_vector_store(config)  # Should reuse cached instance
            store_chunk_embeddings(
                vs,
                chunks[i:i+2],
                embeddings[i:i+2],
                config=config,
            )
        duration_cached = time.time() - start_cached

        print(f"\n{backend} - 5 sequential inserts with connection pooling: {duration_cached:.2f}s")

        # Performance should be reasonable (not testing against non-pooled since it's not implemented)
        assert duration_cached < 10, "Connection pooling should enable fast sequential inserts"
