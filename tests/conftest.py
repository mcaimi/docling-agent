#!/usr/bin/env python
"""Shared pytest fixtures for integration and unit tests."""

import os
import time
import pytest


# Helper to check if Docker services are running
def _wait_for_service(url: str, max_attempts: int = 30, delay: float = 1.0) -> bool:
    """Wait for a service to become available."""
    import requests

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in (200, 404):  # 404 is OK for some endpoints
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


@pytest.fixture(scope="session")
def milvus_available():
    """Check if Milvus is available."""
    if not _wait_for_service("http://localhost:19530", max_attempts=5, delay=1):
        pytest.skip("Milvus not available. Run: cd tests && docker-compose up -d milvus-standalone")
    yield
    # No teardown needed - using session scope


@pytest.fixture(scope="session")
def chromadb_available():
    """Check if ChromaDB is available."""
    if not _wait_for_service("http://localhost:8000/api/v1/heartbeat", max_attempts=5, delay=1):
        pytest.skip("ChromaDB not available. Run: cd tests && docker-compose up -d chromadb")
    yield
    # No teardown needed - using session scope


@pytest.fixture
def milvus_config():
    """Milvus configuration for tests."""
    return {
        "vector_store_backend": "milvus",
        "vector_store_collection": "test_collection",
        "milvus_uri": "http://localhost:19530",
        "milvus_db_name": "default",
        "vector_store_batch_size": 100,
        "vector_store_write_mode": "insert",
        "vector_store_max_batch_workers": 2,
        "vector_store_retry_attempts": 3,
        "vector_store_circuit_breaker_threshold": 5,
        "vector_store_circuit_breaker_timeout": 60,
        "embedding_model": "text-embedding-ada-002",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY", "test-key"),
    }


@pytest.fixture
def chromadb_config():
    """ChromaDB configuration for tests."""
    return {
        "vector_store_backend": "chroma",
        "vector_store_collection": "test_collection",
        "chroma_host": "localhost",
        "chroma_port": 8000,
        "vector_store_batch_size": 100,
        "vector_store_write_mode": "insert",
        "vector_store_max_batch_workers": 2,
        "vector_store_retry_attempts": 3,
        "vector_store_circuit_breaker_threshold": 5,
        "vector_store_circuit_breaker_timeout": 60,
        "embedding_model": "text-embedding-ada-002",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY", "test-key"),
    }


@pytest.fixture
def sample_chunks():
    """Sample chunk data for testing."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {
                "file_path": "/test/doc1.pdf",
                "file_name": "doc1.pdf",
                "chunk_index": 0,
                "chunk_count": 3,
                "page_count": 10,
                "title": "ML Introduction",
            },
        },
        {
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {
                "file_path": "/test/doc1.pdf",
                "file_name": "doc1.pdf",
                "chunk_index": 1,
                "chunk_count": 3,
                "page_count": 10,
                "title": "ML Introduction",
            },
        },
        {
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {
                "file_path": "/test/doc1.pdf",
                "file_name": "doc1.pdf",
                "chunk_index": 2,
                "chunk_count": 3,
                "page_count": 10,
                "title": "ML Introduction",
            },
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings (mock vectors with consistent dimensions)."""
    import random

    random.seed(42)  # Deterministic for testing
    dimensions = 128  # Use smaller dimension for tests
    return [
        [random.random() for _ in range(dimensions)],
        [random.random() for _ in range(dimensions)],
        [random.random() for _ in range(dimensions)],
    ]


@pytest.fixture
def alternative_chunks():
    """Alternative chunks for update/upsert testing."""
    return [
        {
            "content": "Updated content: Machine learning algorithms learn from data.",
            "metadata": {
                "file_path": "/test/doc1.pdf",
                "file_name": "doc1.pdf",
                "chunk_index": 0,
                "chunk_count": 2,
                "page_count": 10,
                "title": "ML Introduction v2",
            },
        },
        {
            "content": "Updated content: Deep learning revolutionized AI in recent years.",
            "metadata": {
                "file_path": "/test/doc1.pdf",
                "file_name": "doc1.pdf",
                "chunk_index": 1,
                "chunk_count": 2,
                "page_count": 10,
                "title": "ML Introduction v2",
            },
        },
    ]


@pytest.fixture
def alternative_embeddings():
    """Alternative embeddings for update/upsert testing."""
    import random

    random.seed(123)  # Different seed for different vectors
    dimensions = 128
    return [
        [random.random() for _ in range(dimensions)],
        [random.random() for _ in range(dimensions)],
    ]
