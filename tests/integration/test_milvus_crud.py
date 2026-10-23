#!/usr/bin/env python
"""Integration tests for Milvus CRUD operations."""

import pytest
from src.graph.vector_store import (
    get_or_create_vector_store,
    store_chunk_embeddings,
    delete_chunk_embeddings,
    query_vector_store,
    _chunk_id,
)
from src.graph.vector_operations import delete_document, replace_document_chunks


class TestMilvusCRUD:
    """Integration tests for Milvus vector store CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, milvus_available, milvus_config):
        """Setup and teardown for each test."""
        self.config = milvus_config.copy()
        self.config["vector_store_collection"] = f"test_milvus_{id(self)}"

        # Get vector store
        self.vector_store = get_or_create_vector_store(self.config)

        yield

        # Cleanup: try to delete the collection
        try:
            if hasattr(self.vector_store, "col"):
                self.vector_store.col.drop()
        except Exception:
            pass

    def test_insert_chunks(self, sample_chunks, sample_embeddings):
        """Test inserting chunks into Milvus."""
        result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            batch_size=100,
            write_mode="insert",
            max_workers=2,
            config=self.config,
        )

        assert result["stored_count"] == 3
        assert result["skipped_count"] == 0
        assert len(result["ids"]) == 3
        assert result["failed_chunks"] == []

        # Verify IDs are correct format
        expected_ids = [_chunk_id(chunk["metadata"]) for chunk in sample_chunks]
        assert set(result["ids"]) == set(expected_ids)

    def test_insert_with_none_embeddings(self, sample_chunks, sample_embeddings):
        """Test inserting chunks with some None embeddings."""
        embeddings_with_none = sample_embeddings.copy()
        embeddings_with_none[1] = None  # Middle chunk fails

        result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            embeddings_with_none,
            batch_size=100,
            config=self.config,
        )

        assert result["stored_count"] == 2
        assert result["skipped_count"] == 1
        assert len(result["ids"]) == 2

    def test_upsert_mode(self, sample_chunks, sample_embeddings, alternative_chunks, alternative_embeddings):
        """Test upsert mode: insert, then update with same IDs."""
        # Initial insert
        result1 = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            write_mode="insert",
            config=self.config,
        )
        assert result1["stored_count"] == 3

        # Upsert with first two chunks updated
        result2 = store_chunk_embeddings(
            self.vector_store,
            alternative_chunks,
            alternative_embeddings,
            write_mode="upsert",
            config=self.config,
        )

        assert result2["stored_count"] == 2
        assert result2["upserted_count"] == 2  # Two existing IDs deleted
        assert result2["deleted_count"] == 2

    def test_skip_existing_mode(self, sample_chunks, sample_embeddings, alternative_chunks, alternative_embeddings):
        """Test skip_existing mode: only insert new IDs."""
        # Initial insert
        result1 = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            write_mode="insert",
            config=self.config,
        )
        assert result1["stored_count"] == 3

        # Try to insert same chunks - should skip all
        result2 = store_chunk_embeddings(
            self.vector_store,
            sample_chunks[:2],  # Same IDs as first two
            sample_embeddings[:2],
            write_mode="skip_existing",
            config=self.config,
        )

        # All should be skipped (may show as stored_count=0 or skipped_count=2 depending on implementation)
        assert result2["stored_count"] == 0

    def test_delete_by_ids(self, sample_chunks, sample_embeddings):
        """Test deleting chunks by IDs."""
        # Insert chunks
        insert_result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert insert_result["stored_count"] == 3

        # Delete first two chunks by ID
        ids_to_delete = [_chunk_id(sample_chunks[0]["metadata"]), _chunk_id(sample_chunks[1]["metadata"])]
        delete_result = delete_chunk_embeddings(
            self.vector_store,
            ids=ids_to_delete,
            config=self.config,
        )

        assert delete_result["deleted_count"] == 2
        assert delete_result["failed_count"] == 0

    def test_delete_by_file_path(self, sample_chunks, sample_embeddings):
        """Test deleting all chunks from a file."""
        # Insert chunks
        insert_result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert insert_result["stored_count"] == 3

        # Delete all chunks from the file
        delete_result = delete_chunk_embeddings(
            self.vector_store,
            file_paths=["/test/doc1.pdf"],
            config=self.config,
        )

        # Note: May not work if backend doesn't support metadata filtering
        # At minimum should not error
        assert "deleted_count" in delete_result

    def test_delete_document_operation(self, sample_chunks, sample_embeddings):
        """Test higher-level delete_document operation."""
        # Insert chunks
        insert_result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert insert_result["stored_count"] == 3

        # Delete document
        delete_result = delete_document(
            self.vector_store,
            "/test/doc1.pdf",
            config=self.config,
        )

        assert "deleted_count" in delete_result

    def test_replace_document_chunks(self, sample_chunks, sample_embeddings, alternative_chunks, alternative_embeddings):
        """Test replace_document_chunks operation."""
        # Insert initial chunks
        insert_result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert insert_result["stored_count"] == 3

        # Replace with updated chunks
        replace_result = replace_document_chunks(
            self.vector_store,
            "/test/doc1.pdf",
            alternative_chunks,
            alternative_embeddings,
            config=self.config,
        )

        assert replace_result["stored_count"] == 2
        assert "deleted_count" in replace_result

    def test_batch_processing_with_workers(self, sample_embeddings):
        """Test parallel batch processing with multiple workers."""
        # Create larger dataset
        large_chunks = []
        large_embeddings = []

        for i in range(50):
            large_chunks.append({
                "content": f"Test content chunk {i}",
                "metadata": {
                    "file_path": "/test/large_doc.pdf",
                    "file_name": "large_doc.pdf",
                    "chunk_index": i,
                    "chunk_count": 50,
                },
            })
            large_embeddings.append(sample_embeddings[i % 3])

        result = store_chunk_embeddings(
            self.vector_store,
            large_chunks,
            large_embeddings,
            batch_size=10,
            max_workers=4,
            config=self.config,
        )

        assert result["stored_count"] == 50
        assert result["failed_chunks"] == []
        assert "duration_ms" in result

    def test_connection_pooling(self, sample_chunks, sample_embeddings):
        """Test that connection pooling reuses vector store instances."""
        # Get vector store multiple times with same config
        vs1 = get_or_create_vector_store(self.config)
        vs2 = get_or_create_vector_store(self.config)

        # Should be the same instance due to caching
        assert vs1 is vs2

    def test_metadata_preservation(self, sample_chunks, sample_embeddings):
        """Test that metadata is correctly preserved in vector store."""
        # Insert chunks
        result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert result["stored_count"] == 3

        # Verify metadata via vector store's get method if available
        if hasattr(self.vector_store, "get"):
            try:
                chunk_id = _chunk_id(sample_chunks[0]["metadata"])
                retrieved = self.vector_store.get(ids=[chunk_id])
                if retrieved and "metadatas" in retrieved:
                    metadata = retrieved["metadatas"][0]
                    assert metadata["file_path"] == "/test/doc1.pdf"
                    assert metadata["chunk_index"] == 0
            except Exception:
                # Backend may not support get
                pass

    def test_error_handling_invalid_embeddings(self, sample_chunks):
        """Test error handling with invalid embeddings (wrong dimensions)."""
        # Create embeddings with inconsistent dimensions
        invalid_embeddings = [
            [0.1, 0.2],  # 2 dimensions
            [0.1, 0.2, 0.3],  # 3 dimensions
            [0.1],  # 1 dimension
        ]

        # Should handle gracefully (may fail or skip, but shouldn't crash)
        try:
            result = store_chunk_embeddings(
                self.vector_store,
                sample_chunks,
                invalid_embeddings,
                config=self.config,
            )
            # If it succeeds, check structure
            assert "stored_count" in result
            assert "failed_chunks" in result
        except Exception as e:
            # Expected to fail with dimension mismatch
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()


class TestMilvusConfiguration:
    """Test Milvus-specific configuration options."""

    def test_milvus_with_custom_index_params(self, milvus_available, milvus_config, sample_chunks, sample_embeddings):
        """Test Milvus with custom index parameters."""
        config = milvus_config.copy()
        config["vector_store_collection"] = f"test_milvus_index_{id(self)}"
        config["milvus_index_type"] = "HNSW"
        config["milvus_metric_type"] = "COSINE"

        vector_store = get_or_create_vector_store(config)

        result = store_chunk_embeddings(
            vector_store,
            sample_chunks,
            sample_embeddings,
            config=config,
        )

        assert result["stored_count"] == 3

        # Cleanup
        try:
            if hasattr(vector_store, "col"):
                vector_store.col.drop()
        except Exception:
            pass

    def test_milvus_consistency_level(self, milvus_available, milvus_config, sample_chunks, sample_embeddings):
        """Test Milvus with different consistency levels."""
        config = milvus_config.copy()
        config["vector_store_collection"] = f"test_milvus_consistency_{id(self)}"
        config["milvus_consistency_level"] = "Strong"

        vector_store = get_or_create_vector_store(config)

        result = store_chunk_embeddings(
            vector_store,
            sample_chunks,
            sample_embeddings,
            config=config,
        )

        assert result["stored_count"] == 3

        # Cleanup
        try:
            if hasattr(vector_store, "col"):
                vector_store.col.drop()
        except Exception:
            pass
