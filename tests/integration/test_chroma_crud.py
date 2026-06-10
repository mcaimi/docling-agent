#!/usr/bin/env python
"""Integration tests for ChromaDB CRUD operations."""

import pytest
from src.graph.vector_store import (
    get_or_create_vector_store,
    store_chunk_embeddings,
    delete_chunk_embeddings,
    query_vector_store,
    _chunk_id,
)
from src.graph.vector_operations import delete_document, replace_document_chunks


class TestChromaDBCRUD:
    """Integration tests for ChromaDB vector store CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, chromadb_available, chromadb_config):
        """Setup and teardown for each test."""
        self.config = chromadb_config.copy()
        self.config["vector_store_collection"] = f"test_chroma_{id(self)}"

        # Get vector store
        self.vector_store = get_or_create_vector_store(self.config)

        yield

        # Cleanup: delete the collection
        try:
            if hasattr(self.vector_store, "_client") and hasattr(self.vector_store._client, "delete_collection"):
                self.vector_store._client.delete_collection(self.config["vector_store_collection"])
        except Exception:
            pass

    def test_insert_chunks(self, sample_chunks, sample_embeddings):
        """Test inserting chunks into ChromaDB."""
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

        # Verify updated content
        if hasattr(self.vector_store, "get"):
            chunk_id = _chunk_id(alternative_chunks[0]["metadata"])
            retrieved = self.vector_store.get(ids=[chunk_id])
            if retrieved and "documents" in retrieved:
                # ChromaDB should have updated content
                assert "Updated content" in retrieved["documents"][0]

    def test_skip_existing_mode(self, sample_chunks, sample_embeddings):
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

        # All should be skipped
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

        # Verify only one chunk remains
        if hasattr(self.vector_store, "get"):
            remaining_id = _chunk_id(sample_chunks[2]["metadata"])
            retrieved = self.vector_store.get(ids=[remaining_id])
            assert retrieved is not None
            assert len(retrieved.get("ids", [])) == 1

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
        """Test that metadata is correctly preserved in ChromaDB."""
        # Insert chunks
        result = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            config=self.config,
        )
        assert result["stored_count"] == 3

        # Verify metadata via get
        chunk_id = _chunk_id(sample_chunks[0]["metadata"])
        retrieved = self.vector_store.get(ids=[chunk_id])

        assert retrieved is not None
        assert "metadatas" in retrieved
        metadata = retrieved["metadatas"][0]
        assert metadata["file_path"] == "/test/doc1.pdf"
        assert metadata["chunk_index"] == 0
        assert metadata["file_name"] == "doc1.pdf"

    def test_chromadb_native_overwrite(self, sample_chunks, sample_embeddings, alternative_embeddings):
        """Test ChromaDB's native overwrite behavior on duplicate IDs."""
        # Insert chunks
        result1 = store_chunk_embeddings(
            self.vector_store,
            sample_chunks,
            sample_embeddings,
            write_mode="insert",
            config=self.config,
        )
        assert result1["stored_count"] == 3

        # Insert again with same IDs but different embeddings (ChromaDB overwrites)
        result2 = store_chunk_embeddings(
            self.vector_store,
            sample_chunks[:1],  # Same ID
            [alternative_embeddings[0]],  # Different embedding
            write_mode="insert",
            config=self.config,
        )

        # ChromaDB should silently overwrite
        assert result2["stored_count"] == 1


class TestChromaDBConfiguration:
    """Test ChromaDB-specific configuration options."""

    def test_chromadb_with_cosine_distance(self, chromadb_available, chromadb_config, sample_chunks, sample_embeddings):
        """Test ChromaDB with cosine distance function."""
        config = chromadb_config.copy()
        config["vector_store_collection"] = f"test_chroma_cosine_{id(self)}"
        config["chroma_distance_function"] = "cosine"

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
            if hasattr(vector_store, "_client"):
                vector_store._client.delete_collection(config["vector_store_collection"])
        except Exception:
            pass

    def test_chromadb_with_l2_distance(self, chromadb_available, chromadb_config, sample_chunks, sample_embeddings):
        """Test ChromaDB with L2 distance function."""
        config = chromadb_config.copy()
        config["vector_store_collection"] = f"test_chroma_l2_{id(self)}"
        config["chroma_distance_function"] = "l2"

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
            if hasattr(vector_store, "_client"):
                vector_store._client.delete_collection(config["vector_store_collection"])
        except Exception:
            pass

    def test_chromadb_collection_metadata(self, chromadb_available, chromadb_config, sample_chunks, sample_embeddings):
        """Test ChromaDB with custom collection metadata."""
        config = chromadb_config.copy()
        config["vector_store_collection"] = f"test_chroma_metadata_{id(self)}"
        config["chroma_collection_metadata"] = {
            "description": "Test collection",
            "created_by": "integration_test",
        }

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
            if hasattr(vector_store, "_client"):
                vector_store._client.delete_collection(config["vector_store_collection"])
        except Exception:
            pass
