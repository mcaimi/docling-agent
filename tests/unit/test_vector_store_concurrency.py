#!/usr/bin/env python
"""Concurrency tests for vector_store's global caches.

document_pipeline runs convert/chunk/embed/store concurrently for every
document in a batch, so get_or_create_vector_store and _get_circuit_breaker
are called from multiple threads at once. Without a lock, the check-then-
create-then-store pattern they use is a TOCTOU race: two threads can both
decide "not cached yet" and each build their own backend connection /
CircuitBreaker for the same key, silently discarding one.
"""

import threading

from src.graph import vector_store as vs


class TestCircuitBreakerRegistryConcurrency:
    def test_one_circuit_breaker_instance_per_backend(self):
        vs._circuit_breakers.clear()
        results: list[tuple[str, int]] = []
        results_lock = threading.Lock()

        def worker(backend: str):
            for _ in range(50):
                cb = vs._get_circuit_breaker(backend, {})
                with results_lock:
                    results.append((backend, id(cb)))

        backends = ["chroma", "milvus"] * 10
        threads = [threading.Thread(target=worker, args=(b,)) for b in backends]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids_by_backend: dict[str, set[int]] = {}
        for backend, obj_id in results:
            ids_by_backend.setdefault(backend, set()).add(obj_id)

        for backend, ids in ids_by_backend.items():
            assert len(ids) == 1, f"{backend} got multiple circuit breaker instances: {ids}"


class TestVectorStoreCacheConcurrency:
    def test_one_instance_per_cache_key_no_cross_contamination(self, monkeypatch):
        construction_count: dict[tuple, int] = {}
        construction_lock = threading.Lock()

        class FakeStore:
            def __init__(self, key):
                self.key = key

        def fake_create_vector_store(config):
            key = vs._build_cache_key(config)
            with construction_lock:
                construction_count[key] = construction_count.get(key, 0) + 1
            return FakeStore(key)

        monkeypatch.setattr(vs, "_create_vector_store", fake_create_vector_store)
        vs._vector_store_cache.clear()
        vs._vector_store_cache_config.clear()

        results: list[tuple[str, tuple]] = []
        results_lock = threading.Lock()

        def worker(collection: str):
            config = {"vector_store_backend": "chroma", "vector_store_collection": collection}
            for _ in range(50):
                store = vs.get_or_create_vector_store(config)
                with results_lock:
                    results.append((collection, store.key))

        collections = ["docs_a", "docs_b"] * 10
        threads = [threading.Thread(target=worker, args=(c,)) for c in collections]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for collection, key in results:
            assert key[1] == collection, f"cross-contamination: {collection} got store for {key}"

        assert all(count == 1 for count in construction_count.values()), construction_count
        assert len(construction_count) == 2
