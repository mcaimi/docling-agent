# Integration Tests for Vector Store CRUD Operations

This directory contains comprehensive integration tests for Milvus and ChromaDB vector store operations.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- UV package manager

## Setup

### 1. Install Test Dependencies

```bash
uv sync --extra dev --extra vector-stores
```

This installs:
- `pytest` and `pytest-asyncio` for testing
- `requests` for health checks
- `langchain-chroma` and `langchain-milvus` for vector store backends

### 2. Start Vector Store Services

From the `tests/` directory, start the Docker containers:

```bash
cd tests
docker-compose up -d
```

This starts:
- **Milvus Standalone** (port 19530) with etcd and MinIO dependencies
- **ChromaDB** (port 8000)

Wait for services to be healthy (~30-60 seconds):

```bash
# Check service health
docker-compose ps

# Or watch logs
docker-compose logs -f
```

Expected output:
```
NAME                 STATUS
milvus-standalone    Up (healthy)
chromadb             Up (healthy)
milvus-etcd          Up (healthy)
milvus-minio         Up (healthy)
```

## Running Tests

### Run All Integration Tests

From the project root:

```bash
uv run pytest tests/integration/ -v
```

### Run Specific Backend Tests

**Milvus only:**
```bash
uv run pytest tests/integration/test_milvus_crud.py -v
```

**ChromaDB only:**
```bash
uv run pytest tests/integration/test_chroma_crud.py -v
```

**Performance tests (both backends):**
```bash
uv run pytest tests/integration/test_vector_store_performance.py -v
```

### Run Unit Tests

```bash
uv run pytest tests/unit/ -v
```

### Run All Tests

```bash
uv run pytest -v
```

## Test Coverage

### Milvus Integration Tests (`test_milvus_crud.py`)

**CRUD Operations:**
- ✅ `test_insert_chunks` - Basic insert operation
- ✅ `test_insert_with_none_embeddings` - Handling None embeddings
- ✅ `test_upsert_mode` - Upsert (delete+insert) behavior
- ✅ `test_skip_existing_mode` - Skip duplicate IDs
- ✅ `test_delete_by_ids` - Delete specific chunks
- ✅ `test_delete_by_file_path` - Delete all chunks from a file
- ✅ `test_delete_document_operation` - Higher-level delete
- ✅ `test_replace_document_chunks` - Document versioning
- ✅ `test_batch_processing_with_workers` - Parallel processing (50 chunks, 4 workers)
- ✅ `test_connection_pooling` - Connection reuse verification
- ✅ `test_metadata_preservation` - Metadata integrity
- ✅ `test_error_handling_invalid_embeddings` - Error handling

**Milvus-Specific Configuration:**
- ✅ `test_milvus_with_custom_index_params` - HNSW index, COSINE metric
- ✅ `test_milvus_consistency_level` - Strong consistency

### ChromaDB Integration Tests (`test_chroma_crud.py`)

**CRUD Operations:**
- ✅ `test_insert_chunks` - Basic insert operation
- ✅ `test_insert_with_none_embeddings` - Handling None embeddings
- ✅ `test_upsert_mode` - Upsert with content verification
- ✅ `test_skip_existing_mode` - Skip duplicate IDs
- ✅ `test_delete_by_ids` - Delete specific chunks with verification
- ✅ `test_delete_by_file_path` - Delete all chunks from a file
- ✅ `test_delete_document_operation` - Higher-level delete
- ✅ `test_replace_document_chunks` - Document versioning
- ✅ `test_batch_processing_with_workers` - Parallel processing (50 chunks, 4 workers)
- ✅ `test_connection_pooling` - Connection reuse verification
- ✅ `test_metadata_preservation` - Metadata integrity with verification
- ✅ `test_chromadb_native_overwrite` - ChromaDB overwrite behavior

**ChromaDB-Specific Configuration:**
- ✅ `test_chromadb_with_cosine_distance` - Cosine distance function
- ✅ `test_chromadb_with_l2_distance` - L2 distance function
- ✅ `test_chromadb_collection_metadata` - Custom collection metadata

### Performance Tests (`test_vector_store_performance.py`)

**Benchmarks (both backends):**
- ✅ `test_large_batch_insert_performance` - 1000 vectors, throughput measurement
- ✅ `test_parallel_batch_performance` - Sequential vs parallel comparison (500 vectors)
- ✅ `test_upsert_performance` - Upsert timing (200 vectors)
- ✅ `test_delete_performance` - Delete timing (250 of 500 vectors)
- ✅ `test_connection_reuse_performance` - Connection pooling benefit

**Performance Expectations:**
- Insert throughput: >50 vectors/sec (reasonable baseline)
- Parallel processing: Should not be slower than sequential
- Connection pooling: <10s for 5 sequential small inserts

### Unit Tests (`tests/unit/`)

**Circuit Breaker (`test_circuit_breaker.py`):**
- ✅ 10 tests covering all states (CLOSED, OPEN, HALF_OPEN)
- ✅ Failure threshold, timeout, recovery, manual reset

## Test Fixtures

### Backend Availability (`conftest.py`)

- `milvus_available` - Checks if Milvus is running (session scope)
- `chromadb_available` - Checks if ChromaDB is running (session scope)

Tests are automatically skipped if backends are not available with helpful error messages.

### Configuration Fixtures

- `milvus_config` - Standard Milvus configuration
- `chromadb_config` - Standard ChromaDB configuration

### Data Fixtures

- `sample_chunks` - 3 sample chunks with metadata
- `sample_embeddings` - 3 mock 128-dimensional vectors (deterministic)
- `alternative_chunks` - 2 updated chunks for upsert testing
- `alternative_embeddings` - 2 alternative vectors

## Cleanup

Tests automatically clean up collections after each test using `autouse=True` fixtures.

### Manual Cleanup

If tests are interrupted, you may need to manually clean up:

**Milvus:**
```python
from pymilvus import connections, utility
connections.connect(host="localhost", port="19530")
utility.drop_collection("test_collection_name")
```

**ChromaDB:**
```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
client.delete_collection("test_collection_name")
```

## Stopping Services

```bash
cd tests
docker-compose down

# Remove volumes (complete cleanup)
docker-compose down -v
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose logs milvus-standalone
docker-compose logs chromadb

# Restart services
docker-compose restart
```

### Tests Skip with "Service Not Available"

1. Verify services are running: `docker-compose ps`
2. Check health: `curl http://localhost:19530` (Milvus) and `curl http://localhost:8000/api/v1/heartbeat` (ChromaDB)
3. Wait longer for services to become healthy (especially Milvus on first start)

### Port Conflicts

If ports 19530 or 8000 are already in use:

1. Edit `docker-compose.yml` to use different ports
2. Update corresponding fixtures in `conftest.py`

### Permission Errors

```bash
# Ensure Docker has permissions
sudo docker-compose up -d
```

## CI/CD Integration

To run these tests in CI:

```yaml
# Example GitHub Actions workflow
- name: Start vector stores
  run: |
    cd tests
    docker-compose up -d
    sleep 30  # Wait for services

- name: Run integration tests
  run: |
    uv sync --extra dev --extra vector-stores
    uv run pytest tests/integration/ -v

- name: Stop services
  run: cd tests && docker-compose down -v
```

## Test Metrics

Running all integration tests typically takes:
- **Milvus tests**: ~30-60 seconds
- **ChromaDB tests**: ~20-40 seconds  
- **Performance tests**: ~60-120 seconds (includes both backends)
- **Total**: ~2-4 minutes

Actual times depend on hardware and Docker performance.
