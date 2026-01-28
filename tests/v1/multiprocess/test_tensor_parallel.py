# SPDX-License-Identifier: Apache-2.0
"""
Tests for tensor parallel (TP) support in the multiprocess cache engine.

This module tests the TP lookup mechanism where:
- scheduler uses worker_id=None to lookup cache across all workers
- workers use specific worker_id for store/retrieve operations
- lookup requires ALL workers to have the cache for a hit

Key scenarios tested:
- TP=2 with both workers having all chunks cached
- TP=2 with only one worker having cache (asymmetric)
- TP=2 with different partial hits across workers
- Various world sizes (TP=1, TP=2, TP=4, TP=8)
"""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.multiprocess.custom_types import (
    StorageKey,
)
from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def storage_manager():
    """Create a storage manager with 1GB buffer for testing."""
    manager = MPStorageManager(cpu_buffer_size=1.0)
    yield manager
    manager.close()


@pytest.fixture
def test_shape():
    """Standard test shape for tensors."""
    return torch.Size((2, 16, 16, 128))


@pytest.fixture
def test_dtype():
    """Standard test dtype for tensors."""
    return torch.float16


@pytest.fixture
def test_format():
    """Standard test memory format."""
    return MemoryFormat.KV_2LTD


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_storage_key(
    chunk_hash: int,
    worker_id: int,
    world_size: int = 2,
    model_name: str = "test_model",
) -> StorageKey:
    """Create a StorageKey for testing."""
    return StorageKey.from_int_hash(
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def create_interleaved_lookup_keys(
    num_chunks: int,
    world_size: int,
    model_name: str = "test_model",
) -> list[StorageKey]:
    """
    Create interleaved lookup keys for scheduler-style TP lookup.

    The order matches what the scheduler expects:
    [chunk0_worker0, chunk0_worker1, ..., chunk0_workerN,
     chunk1_worker0, chunk1_worker1, ..., chunk1_workerN, ...]

    This simulates the key expansion that happens for scheduler lookups
    where worker_id=None gets expanded to all workers.
    """
    keys = []
    for chunk_idx in range(num_chunks):
        for worker_id in range(world_size):
            keys.append(
                create_storage_key(
                    chunk_hash=chunk_idx,
                    worker_id=worker_id,
                    world_size=world_size,
                    model_name=model_name,
                )
            )
    return keys


# ==============================================================================
# Tests for Storage Manager with TP Scenarios
# ==============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for tensor parallel tests",
)
class TestStorageManagerTPLookup:
    """
    Tests for storage manager lookup with tensor parallel scenarios.

    The key invariant: for a scheduler lookup (worker_id=None) to succeed,
    ALL workers must have the cache stored for that chunk.
    """

    def test_tp2_both_workers_have_all_chunks(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=2 lookup when both workers have all chunks cached.
        Expected: All lookups return True.
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for both workers
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Create interleaved lookup keys for scheduler-style lookup
        lookup_keys = create_interleaved_lookup_keys(num_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # All keys should be found (5 chunks * 2 workers = 10)
        assert found_count == num_chunks * world_size

        # Simulating MPCacheEngine.lookup logic
        found_ipc_count = found_count // world_size
        assert found_ipc_count == num_chunks

    def test_tp2_only_worker0_has_cache_asymmetric(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=2 lookup when only worker 0 has cache (asymmetric).
        Expected: Lookup returns 0 (no complete cache hit).
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for worker 0 only
        storage_keys = [
            create_storage_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(num_chunks)
        ]
        handle, _ = storage_manager.reserve(
            storage_keys, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Create interleaved lookup keys for scheduler-style lookup
        lookup_keys = create_interleaved_lookup_keys(num_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # Only worker 0's first chunk is found, then lookup stops
        # at worker 1's missing chunk
        # The ordering is: [chunk0_worker0, chunk0_worker1, chunk1_worker0, ...]
        # So we find chunk0_worker0 (1), then miss chunk0_worker1
        assert found_count == 1

        # Simulating MPCacheEngine.lookup logic
        found_ipc_count = found_count // world_size
        # 1 // 2 = 0, so no complete cache hit
        assert found_ipc_count == 0

    def test_tp2_only_worker1_has_cache_asymmetric(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=2 lookup when only worker 1 has cache (asymmetric).
        Expected: Lookup returns 0 (first key for worker 0 is missing).
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for worker 1 only
        storage_keys = [
            create_storage_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(num_chunks)
        ]
        handle, _ = storage_manager.reserve(
            storage_keys, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Create interleaved lookup keys for scheduler-style lookup
        lookup_keys = create_interleaved_lookup_keys(num_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # First lookup key is chunk0_worker0 which is missing
        assert found_count == 0

        # Simulating MPCacheEngine.lookup logic
        found_ipc_count = found_count // world_size
        assert found_ipc_count == 0

    def test_tp2_partial_prefix_both_workers(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=2 lookup with partial prefix: both workers have first 3 chunks.
        Expected: First 3 chunks return True, rest return False.
        """
        world_size = 2
        num_stored_chunks = 3
        num_requested_chunks = 5

        # Store first 3 chunks for both workers
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_stored_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Request 5 chunks with scheduler-style interleaved lookup
        lookup_keys = create_interleaved_lookup_keys(num_requested_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # First 3 chunks * 2 workers = 6 keys found, then stops at chunk3_worker0
        assert found_count == num_stored_chunks * world_size

        # Simulating MPCacheEngine.lookup logic
        found_ipc_count = found_count // world_size
        assert found_ipc_count == num_stored_chunks

    def test_tp2_different_partial_hits_min_common_prefix(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=2 with different partial hits across workers.
        Worker 0: has chunks 0, 1, 2, 3, 4 (5 chunks)
        Worker 1: has chunks 0, 1 (2 chunks)
        Expected: Only first 2 chunks are counted (minimum common prefix).
        """
        world_size = 2

        # Worker 0 has 5 chunks
        storage_keys_w0 = [
            create_storage_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(5)
        ]
        handle, _ = storage_manager.reserve(
            storage_keys_w0, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Worker 1 has only 2 chunks
        storage_keys_w1 = [
            create_storage_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(2)
        ]
        handle, _ = storage_manager.reserve(
            storage_keys_w1, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Request 5 chunks with scheduler-style interleaved lookup
        lookup_keys = create_interleaved_lookup_keys(5, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # Lookup order:
        # chunk0_w0, chunk0_w1, chunk1_w0, chunk1_w1, chunk2_w0, chunk2_w1...
        # chunk0_w0: found (1)
        # chunk0_w1: found (2)
        # chunk1_w0: found (3)
        # chunk1_w1: found (4)
        # chunk2_w0: found (5)
        # chunk2_w1: NOT found (stops)
        assert found_count == 5  # 2 complete chunks * 2 workers + 1 partial

        # Simulating MPCacheEngine.lookup logic
        found_ipc_count = found_count // world_size
        # 5 // 2 = 2, so only 2 complete chunks
        assert found_ipc_count == 2

    def test_tp4_all_workers_have_cache(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=4 lookup when all 4 workers have all chunks cached.
        """
        world_size = 4
        num_chunks = 3

        # Store chunks for all workers
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Scheduler-style interleaved lookup
        lookup_keys = create_interleaved_lookup_keys(num_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # All keys found: 3 chunks * 4 workers = 12
        assert found_count == num_chunks * world_size

        found_ipc_count = found_count // world_size
        assert found_ipc_count == num_chunks

    def test_tp4_one_worker_missing_causes_no_hit(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test TP=4 where one worker (worker 2) is missing all cache.
        Expected: No complete hits due to prefix matching.
        """
        world_size = 4
        num_chunks = 3

        # Store chunks for workers 0, 1, 3 (skip worker 2)
        for worker_id in [0, 1, 3]:
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Scheduler-style interleaved lookup
        lookup_keys = create_interleaved_lookup_keys(num_chunks, world_size)
        found_count = storage_manager.lookup(lookup_keys)

        # Lookup order: chunk0_w0, chunk0_w1, chunk0_w2, chunk0_w3, ...
        # chunk0_w0: found (1)
        # chunk0_w1: found (2)
        # chunk0_w2: NOT found (stops)
        assert found_count == 2

        found_ipc_count = found_count // world_size
        # 2 // 4 = 0, no complete chunks
        assert found_ipc_count == 0


# ==============================================================================
# Tests for Store and Retrieve with TP
# ==============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for tensor parallel tests",
)
class TestStorageManagerTPStoreRetrieve:
    """Tests for store and retrieve operations with tensor parallel."""

    def test_tp2_store_creates_separate_keys(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test that storing with different worker_ids creates separate entries.
        """
        world_size = 2

        # Store same chunk hash but different worker_ids
        key_w0 = create_storage_key(chunk_hash=100, worker_id=0, world_size=world_size)
        key_w1 = create_storage_key(chunk_hash=100, worker_id=1, world_size=world_size)

        # Store worker 0's data
        handle0, reserved0 = storage_manager.reserve(
            [key_w0], test_shape, test_dtype, test_format
        )
        assert len(reserved0) == 1
        storage_manager.commit(handle0)

        # Store worker 1's data
        handle1, reserved1 = storage_manager.reserve(
            [key_w1], test_shape, test_dtype, test_format
        )
        assert len(reserved1) == 1
        storage_manager.commit(handle1)

        # Both should be retrievable independently
        with storage_manager.retrieve([key_w0]) as objs:
            assert len(objs) == 1

        with storage_manager.retrieve([key_w1]) as objs:
            assert len(objs) == 1

    def test_tp2_retrieve_specific_worker(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test that retrieve with specific worker_id only gets that worker's data.
        """
        world_size = 2

        # Store for both workers
        for worker_id in range(world_size):
            keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(3)
            ]
            handle, _ = storage_manager.reserve(
                keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Retrieve only worker 0's data
        keys_w0 = [
            create_storage_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(3)
        ]
        with storage_manager.retrieve(keys_w0) as objs:
            assert len(objs) == 3

        # Retrieve only worker 1's data
        keys_w1 = [
            create_storage_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(3)
        ]
        with storage_manager.retrieve(keys_w1) as objs:
            assert len(objs) == 3


# ==============================================================================
# Tests for Edge Cases
# ==============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for tensor parallel tests",
)
class TestTPEdgeCases:
    """Edge case tests for tensor parallel support."""

    def test_world_size_1_stores_and_retrieves(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test that world_size=1 (no TP) works correctly through the API.
        Single worker stores and retrieves data successfully.
        """
        world_size = 1
        num_chunks = 3

        # Store chunks for worker 0
        storage_keys = [
            create_storage_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(num_chunks)
        ]
        handle, _ = storage_manager.reserve(
            storage_keys, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Lookup should find all chunks
        found_count = storage_manager.lookup(storage_keys)
        assert found_count == num_chunks

        # Retrieve should work
        with storage_manager.retrieve(storage_keys) as objs:
            assert len(objs) == num_chunks

    def test_large_world_size_tp8(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test with larger world_size (TP=8) through the API.
        All 8 workers store and lookup works correctly.
        """
        world_size = 8
        num_chunks = 3

        # Store chunks for all workers
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Create interleaved lookup keys (simulating scheduler lookup)
        # Order: [chunk0_w0, chunk0_w1, ..., chunk0_w7, chunk1_w0, ...]
        lookup_keys = []
        for chunk_idx in range(num_chunks):
            for worker_id in range(world_size):
                lookup_keys.append(
                    create_storage_key(
                        chunk_hash=chunk_idx, worker_id=worker_id, world_size=world_size
                    )
                )

        # All keys should be found
        found_count = storage_manager.lookup(lookup_keys)
        assert found_count == num_chunks * world_size

        # Verify retrieval for each worker
        for worker_id in range(world_size):
            worker_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            with storage_manager.retrieve(worker_keys) as objs:
                assert len(objs) == num_chunks

    def test_all_workers_same_chunk_different_keys(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test that same chunk_hash with different worker_ids creates
        distinct entries in storage and can be stored/retrieved independently.
        """
        world_size = 4
        chunk_hash = 42

        # Create storage keys for all workers with same chunk_hash
        storage_keys = [
            create_storage_key(
                chunk_hash=chunk_hash, worker_id=i, world_size=world_size
            )
            for i in range(world_size)
        ]

        # All keys should be distinct
        assert len(set(storage_keys)) == world_size

        # Store all keys
        handle, reserved = storage_manager.reserve(
            storage_keys, test_shape, test_dtype, test_format
        )
        assert len(reserved) == world_size
        storage_manager.commit(handle)

        # Lookup all keys
        found_count = storage_manager.lookup(storage_keys)
        assert found_count == world_size

        # Retrieve each worker's key independently
        for worker_id in range(world_size):
            worker_key = create_storage_key(
                chunk_hash=chunk_hash, worker_id=worker_id, world_size=world_size
            )
            with storage_manager.retrieve([worker_key]) as objs:
                assert len(objs) == 1
                assert objs[0] is not None


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for tensor parallel tests",
)
class TestTPIntegration:
    """Integration tests simulating real TP workflows."""

    def test_full_tp2_workflow(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Simulate a full TP=2 workflow:
        1. Worker 0 stores chunks 0, 1, 2
        2. Worker 1 stores chunks 0, 1, 2
        3. Scheduler looks up chunks 0, 1, 2, 3, 4 (interleaved for all workers)
        4. Verify correct hit count
        5. Workers retrieve their respective chunks
        """
        world_size = 2
        stored_chunks = 3
        requested_chunks = 5

        # Step 1 & 2: Workers store their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(stored_chunks)
            ]
            handle, reserved = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            assert len(reserved) == stored_chunks
            storage_manager.commit(handle)

        # Step 3: Scheduler lookup with interleaved keys
        # Order: [chunk0_w0, chunk0_w1, chunk1_w0, chunk1_w1, ...]
        lookup_keys = []
        for chunk_idx in range(requested_chunks):
            for worker_id in range(world_size):
                lookup_keys.append(
                    create_storage_key(
                        chunk_hash=chunk_idx,
                        worker_id=worker_id,
                        world_size=world_size,
                    )
                )
        found_count = storage_manager.lookup(lookup_keys)

        # Step 4: Verify hit count
        # First 3 chunks * 2 workers = 6 keys found, then stops at chunk3_worker0
        assert found_count == stored_chunks * world_size

        # Compute number of complete IPC-level hits
        found_ipc_count = found_count // world_size
        assert found_ipc_count == stored_chunks

        # Step 5: Workers retrieve their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(stored_chunks)
            ]
            with storage_manager.retrieve(storage_keys) as objs:
                assert len(objs) == stored_chunks
                for obj in objs:
                    assert obj is not None

    def test_concurrent_tp2_stores(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Test concurrent stores from multiple "workers" (threads).
        """
        world_size = 2
        num_chunks = 10
        results = {}

        def worker_store(worker_id: int):
            storage_keys = [
                create_storage_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            handle, reserved = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)
            results[worker_id] = len(reserved)

        # Run stores concurrently
        threads = []
        for worker_id in range(world_size):
            t = threading.Thread(target=worker_store, args=(worker_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify both workers stored their chunks
        assert results[0] == num_chunks
        assert results[1] == num_chunks

        # Verify lookup works with interleaved keys
        lookup_keys = []
        for chunk_idx in range(num_chunks):
            for worker_id in range(world_size):
                lookup_keys.append(
                    create_storage_key(
                        chunk_hash=chunk_idx,
                        worker_id=worker_id,
                        world_size=world_size,
                    )
                )
        found_count = storage_manager.lookup(lookup_keys)
        assert found_count == num_chunks * world_size
