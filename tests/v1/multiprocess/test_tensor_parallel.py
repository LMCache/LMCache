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
- The ipc_keys_to_storage_keys conversion function
"""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey, StorageKey
from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager
from lmcache.v1.multiprocess.server import ipc_keys_to_storage_keys


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
    return (2, 16, 16, 128)


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


def create_ipc_key(
    chunk_hash: int,
    worker_id: int | None = None,
    world_size: int = 2,
    model_name: str = "test_model",
) -> IPCCacheEngineKey:
    """Create an IPCCacheEngineKey for testing."""
    return IPCCacheEngineKey.from_int_hash(
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


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


# ==============================================================================
# Tests for ipc_keys_to_storage_keys function
# ==============================================================================


class TestIpcKeysToStorageKeys:
    """Tests for the ipc_keys_to_storage_keys conversion function."""

    def test_empty_keys(self):
        """Test conversion with empty key list."""
        result = ipc_keys_to_storage_keys([])
        assert result == []

    def test_single_key_with_worker_id(self):
        """Test conversion of a single key with specific worker_id."""
        ipc_key = create_ipc_key(chunk_hash=100, worker_id=0, world_size=2)
        result = ipc_keys_to_storage_keys([ipc_key])

        assert len(result) == 1
        assert result[0].worker_id == 0
        assert result[0].chunk_hash == ipc_key.chunk_hash

    def test_single_key_with_none_worker_id_tp2(self):
        """Test conversion of a single key with worker_id=None (TP=2)."""
        ipc_key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=2)
        result = ipc_keys_to_storage_keys([ipc_key])

        assert len(result) == 2
        assert result[0].worker_id == 0
        assert result[1].worker_id == 1
        # Both should have the same chunk_hash
        assert result[0].chunk_hash == ipc_key.chunk_hash
        assert result[1].chunk_hash == ipc_key.chunk_hash

    def test_single_key_with_none_worker_id_tp4(self):
        """Test conversion of a single key with worker_id=None (TP=4)."""
        ipc_key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=4)
        result = ipc_keys_to_storage_keys([ipc_key])

        assert len(result) == 4
        for i in range(4):
            assert result[i].worker_id == i
            assert result[i].chunk_hash == ipc_key.chunk_hash

    def test_multiple_keys_with_none_worker_id(self):
        """Test conversion of multiple keys with worker_id=None (TP=2)."""
        ipc_keys = [
            create_ipc_key(chunk_hash=100, worker_id=None, world_size=2),
            create_ipc_key(chunk_hash=101, worker_id=None, world_size=2),
            create_ipc_key(chunk_hash=102, worker_id=None, world_size=2),
        ]
        result = ipc_keys_to_storage_keys(ipc_keys)

        # 3 IPC keys * 2 workers = 6 storage keys
        assert len(result) == 6

        # Check ordering: [chunk0_worker0, chunk0_worker1, chunk1_worker0, ...]
        expected_order = [
            (100, 0),
            (100, 1),
            (101, 0),
            (101, 1),
            (102, 0),
            (102, 1),
        ]
        for i, (chunk_hash, worker_id) in enumerate(expected_order):
            assert StorageKey.Bytes2IntHash(result[i].chunk_hash) == chunk_hash
            assert result[i].worker_id == worker_id

    def test_mixed_keys_with_and_without_worker_id(self):
        """Test conversion of mixed keys (some with worker_id, some without)."""
        ipc_keys = [
            create_ipc_key(chunk_hash=100, worker_id=0, world_size=2),
            create_ipc_key(chunk_hash=101, worker_id=None, world_size=2),
        ]
        result = ipc_keys_to_storage_keys(ipc_keys)

        # First key: 1 storage key (worker_id=0)
        # Second key: 2 storage keys (worker_id=0, 1)
        assert len(result) == 3

        assert result[0].worker_id == 0
        assert StorageKey.Bytes2IntHash(result[0].chunk_hash) == 100

        assert result[1].worker_id == 0
        assert StorageKey.Bytes2IntHash(result[1].chunk_hash) == 101

        assert result[2].worker_id == 1
        assert StorageKey.Bytes2IntHash(result[2].chunk_hash) == 101

    def test_inconsistent_world_size_raises_error(self):
        """Test that inconsistent world_size values raise ValueError."""
        ipc_keys = [
            create_ipc_key(chunk_hash=100, worker_id=None, world_size=2),
            create_ipc_key(chunk_hash=101, worker_id=None, world_size=4),
        ]
        with pytest.raises(ValueError, match="same world_size"):
            ipc_keys_to_storage_keys(ipc_keys)


# ==============================================================================
# Tests for IPCCacheEngineKey.no_worker_id_version()
# ==============================================================================


class TestIPCCacheEngineKeyNoWorkerIdVersion:
    """Tests for the no_worker_id_version() method."""

    def test_converts_worker_id_to_none(self):
        """Test that no_worker_id_version converts worker_id to None."""
        key = create_ipc_key(chunk_hash=100, worker_id=1, world_size=2)
        no_worker_key = key.no_worker_id_version()

        assert no_worker_key.worker_id is None
        assert no_worker_key.model_name == key.model_name
        assert no_worker_key.world_size == key.world_size
        assert no_worker_key.chunk_hash == key.chunk_hash

    def test_already_none_remains_none(self):
        """Test that a key with worker_id=None remains unchanged."""
        key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=2)
        no_worker_key = key.no_worker_id_version()

        assert no_worker_key.worker_id is None
        assert no_worker_key == key


# ==============================================================================
# Tests for Storage Manager with TP Scenarios
# ==============================================================================


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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Create IPC keys for scheduler lookup (worker_id=None)
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]

        # Convert to storage keys and lookup
        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
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

        # Create IPC keys for scheduler lookup (worker_id=None)
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]

        # Convert to storage keys and lookup
        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
        found_count = storage_manager.lookup(lookup_keys)

        # Only worker 0's first chunk is found, then lookup stops at worker 1's missing chunk
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

        # Create IPC keys for scheduler lookup (worker_id=None)
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]

        # Convert to storage keys and lookup
        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(num_stored_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Request 5 chunks with scheduler lookup
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_requested_chunks)
        ]

        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
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

        # Request 5 chunks with scheduler lookup
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(5)
        ]

        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
        found_count = storage_manager.lookup(lookup_keys)

        # Lookup order: chunk0_w0, chunk0_w1, chunk1_w0, chunk1_w1, chunk2_w0, chunk2_w1...
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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Scheduler lookup
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]

        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(num_chunks)
            ]
            handle, _ = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            storage_manager.commit(handle)

        # Scheduler lookup
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]

        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(3)
            ]
            handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)
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


class TestTPEdgeCases:
    """Edge case tests for tensor parallel support."""

    def test_world_size_1_no_expansion(self):
        """Test that world_size=1 (no TP) doesn't expand keys unnecessarily."""
        ipc_key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=1)
        result = ipc_keys_to_storage_keys([ipc_key])

        assert len(result) == 1
        assert result[0].worker_id == 0

    def test_large_world_size(self):
        """Test with larger world_size (TP=8)."""
        world_size = 8
        ipc_key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=world_size)
        result = ipc_keys_to_storage_keys([ipc_key])

        assert len(result) == world_size
        for i in range(world_size):
            assert result[i].worker_id == i

    def test_storage_key_requires_worker_id(self):
        """Test that StorageKey always has integer worker_id (not None)."""
        key = create_storage_key(chunk_hash=100, worker_id=0, world_size=2)
        assert isinstance(key.worker_id, int)
        assert key.worker_id == 0

    def test_ipc_key_serialization_with_none_worker_id(self):
        """Test that IPCCacheEngineKey with worker_id=None serializes correctly."""
        key = create_ipc_key(chunk_hash=100, worker_id=None, world_size=2)

        # Serialize
        encoded = IPCCacheEngineKey.Serialize(key)

        # Deserialize
        decoded = IPCCacheEngineKey.Deserialize(encoded)

        assert decoded.worker_id is None
        assert decoded.world_size == key.world_size
        assert decoded.chunk_hash == key.chunk_hash

    def test_ipc_key_serialization_with_int_worker_id(self):
        """Test that IPCCacheEngineKey with integer worker_id serializes correctly."""
        key = create_ipc_key(chunk_hash=100, worker_id=1, world_size=2)

        # Serialize
        encoded = IPCCacheEngineKey.Serialize(key)

        # Deserialize
        decoded = IPCCacheEngineKey.Deserialize(encoded)

        assert decoded.worker_id == 1
        assert decoded.world_size == key.world_size
        assert decoded.chunk_hash == key.chunk_hash

    def test_all_workers_same_chunk_different_keys(self):
        """Test that same chunk_hash with different worker_ids creates distinct storage keys."""
        world_size = 4
        chunk_hash = 42

        storage_keys = [
            create_storage_key(chunk_hash=chunk_hash, worker_id=i, world_size=world_size)
            for i in range(world_size)
        ]

        # All keys should be distinct
        assert len(set(storage_keys)) == world_size

        # But all share the same chunk_hash
        for key in storage_keys:
            assert StorageKey.Bytes2IntHash(key.chunk_hash) == chunk_hash


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestTPIntegration:
    """Integration tests simulating real TP workflows."""

    def test_full_tp2_workflow(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """
        Simulate a full TP=2 workflow:
        1. Worker 0 stores chunks 0, 1, 2
        2. Worker 1 stores chunks 0, 1, 2
        3. Scheduler looks up chunks 0, 1, 2, 3, 4
        4. Verify correct hit count
        5. Workers retrieve their respective chunks
        """
        world_size = 2
        stored_chunks = 3
        requested_chunks = 5

        # Step 1 & 2: Workers store their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
                for i in range(stored_chunks)
            ]
            handle, reserved = storage_manager.reserve(
                storage_keys, test_shape, test_dtype, test_format
            )
            assert len(reserved) == stored_chunks
            storage_manager.commit(handle)

        # Step 3: Scheduler lookup
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(requested_chunks)
        ]
        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
        found_count = storage_manager.lookup(lookup_keys)

        # Step 4: Verify hit count
        found_ipc_count = found_count // world_size
        assert found_ipc_count == stored_chunks
        expected_result = [True] * stored_chunks + [False] * (requested_chunks - stored_chunks)
        actual_result = [True] * found_ipc_count + [False] * (requested_chunks - found_ipc_count)
        assert actual_result == expected_result

        # Step 5: Workers retrieve their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
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
                create_storage_key(chunk_hash=i, worker_id=worker_id, world_size=world_size)
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

        # Verify lookup works
        ipc_keys = [
            create_ipc_key(chunk_hash=i, worker_id=None, world_size=world_size)
            for i in range(num_chunks)
        ]
        lookup_keys = ipc_keys_to_storage_keys(ipc_keys)
        found_count = storage_manager.lookup(lookup_keys)
        assert found_count == num_chunks * world_size
