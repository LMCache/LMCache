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
from lmcache import torch_dev
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
)
from lmcache.v1.distributed.bitmap_ops import fold_unfold_grouped
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryFormat

# Test helpers
from tests.v1.distributed.utils import ranked_spec, single_row_spec

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def storage_manager():
    """Create a storage manager with 1GB buffer for testing."""
    l1_memory_config = L1MemoryManagerConfig(
        size_in_bytes=1 << 30,
        use_lazy=True,
        init_size_in_bytes=1 << 30,
    )
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=l1_memory_config,
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
    )
    manager = StorageManager(config=storage_manager_config)
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
def test_layout(test_shape, test_dtype):
    return MemoryLayoutDesc(
        shapes=[test_shape],
        dtypes=[test_dtype],
    )


@pytest.fixture
def test_format():
    """Standard test memory format."""
    return MemoryFormat.KV_2LTD


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_object_key(
    chunk_hash: int,
    worker_id: int,
    world_size: int = 2,
    model_name: str = "test_model",
) -> ObjectKey:
    """Create an ObjectKey for testing."""
    kv_rank = ObjectKey.ComputeKVRank(
        world_size=world_size,
        global_rank=worker_id,
        local_world_size=world_size,
        local_rank=worker_id,
    )
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_hash),
        model_name=model_name,
        kv_rank=kv_rank,
    )


def create_lookup_key_rows(
    num_chunks: int,
    world_size: int,
    model_name: str = "test_model",
) -> list[list[ObjectKey]]:
    """
    Create the per-rank key rows of a scheduler-style TP lookup.

    Row ``r`` holds worker ``r``'s keys for chunks ``0..num_chunks-1`` in
    chunk order -- the ``GroupedObjectKeys`` layout the scheduler submits when a
    lookup with ``worker_id=None`` is expanded to all workers.
    """
    return [
        [
            create_object_key(
                chunk_hash=chunk_idx,
                worker_id=worker_id,
                world_size=world_size,
                model_name=model_name,
            )
            for chunk_idx in range(num_chunks)
        ]
        for worker_id in range(world_size)
    ]


def tp_lookup_hit_chunks(
    storage_manager: StorageManager, handle: PrefetchHandle, world_size: int
) -> tuple[int, list]:
    """
    Resolve a TP lookup the way ``MPCacheServer.lookup`` does.

    Returns ``(hit_chunks, rows)``: the number of leading chunks every worker
    has (the fold over the per-rank rows) and the per-rank found bitmaps.
    """
    rows = storage_manager.query_prefetch_status(handle)
    assert rows is not None
    hit_chunks, _ = fold_unfold_grouped(rows, [-1] * world_size)
    return hit_chunks, rows


# ==============================================================================
# Tests for Storage Manager with TP Scenarios
# ==============================================================================


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
class TestStorageManagerTPLookup:
    """
    Tests for storage manager lookup with tensor parallel scenarios.

    The key invariant: for a scheduler lookup (worker_id=None) to succeed,
    ALL workers must have the cache stored for that chunk.
    """

    def test_tp2_both_workers_have_all_chunks(self, storage_manager, test_layout):
        """
        Test TP=2 lookup when both workers have all chunks cached.
        Expected: All lookups return True.
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for both workers
        for worker_id in range(world_size):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler-style lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Every worker has every chunk: all 5 chunks hit.
        assert hit_chunks == num_chunks
        for row in found_rows:
            assert row.get_indices_list() == list(range(num_chunks))

    def test_tp2_only_worker0_has_cache_asymmetric(self, storage_manager, test_layout):
        """
        Test TP=2 lookup when only worker 0 has cache (asymmetric).
        Expected: Lookup returns 0 (no complete cache hit).
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for worker 0 only
        storage_keys = [
            create_object_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(num_chunks)
        ]
        reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler-style lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # A chunk hits only when every worker has it; worker 1 has none, so
        # nothing hits and worker 0's out-of-prefix keys are released.
        assert hit_chunks == 0
        for row in found_rows:
            assert row.popcount() == 0

    def test_tp2_only_worker1_has_cache_asymmetric(self, storage_manager, test_layout):
        """
        Test TP=2 lookup when only worker 1 has cache (asymmetric).
        Expected: Lookup returns 0 (first key for worker 0 is missing).
        """
        world_size = 2
        num_chunks = 5

        # Store chunks for worker 1 only
        storage_keys = [
            create_object_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(num_chunks)
        ]
        reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler-style lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Worker 0 has nothing, so no chunk is complete.
        assert hit_chunks == 0
        for row in found_rows:
            assert row.popcount() == 0

    def test_tp2_partial_prefix_both_workers(self, storage_manager, test_layout):
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
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_stored_chunks)
            ]
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Request 5 chunks with a scheduler-style lookup (one row per worker)
        rows = create_lookup_key_rows(num_requested_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Both workers have chunks 0-2: a 3-chunk hit.
        assert hit_chunks == num_stored_chunks
        for row in found_rows:
            assert row.get_indices_list() == list(range(num_stored_chunks))

    def test_tp2_different_partial_hits_min_common_prefix(
        self, storage_manager, test_layout
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
            create_object_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(5)
        ]
        reserved_dict = storage_manager.reserve_write(storage_keys_w0, test_layout)
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Worker 1 has only 2 chunks
        storage_keys_w1 = [
            create_object_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(2)
        ]
        reserved_dict = storage_manager.reserve_write(storage_keys_w1, test_layout)
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Request 5 chunks with a scheduler-style lookup (one row per worker)
        rows = create_lookup_key_rows(5, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Worker 0: chunks 0-4 resident; worker 1: chunks 0-1. The hit is the
        # minimum common prefix (2 chunks); worker 0's chunks 2-4 are released.
        assert hit_chunks == 2
        assert found_rows[0].get_indices_list() == [0, 1]
        assert found_rows[1].get_indices_list() == [0, 1]

    def test_tp4_all_workers_have_cache(self, storage_manager, test_layout):
        """
        Test TP=4 lookup when all 4 workers have all chunks cached.
        """
        world_size = 4
        num_chunks = 3

        # Store chunks for all workers
        for worker_id in range(world_size):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler-style lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # All 4 workers have all 3 chunks.
        assert hit_chunks == num_chunks
        for row in found_rows:
            assert row.get_indices_list() == list(range(num_chunks))

    def test_tp4_one_worker_missing_causes_no_hit(self, storage_manager, test_layout):
        """
        Test TP=4 where one worker (worker 2) is missing all cache.
        Expected: No complete hits due to prefix matching.
        """
        world_size = 4
        num_chunks = 3

        # Store chunks for workers 0, 1, 3 (skip worker 2)
        for worker_id in [0, 1, 3]:
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler-style lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Worker 2 has nothing, so no chunk is complete across all workers.
        assert hit_chunks == 0
        for row in found_rows:
            assert row.popcount() == 0


# ==============================================================================
# Tests for Store and Retrieve with TP
# ==============================================================================


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
class TestStorageManagerTPStoreRetrieve:
    """Tests for store and retrieve operations with tensor parallel."""

    def test_tp2_store_creates_separate_keys(self, storage_manager, test_layout):
        """
        Test that storing with different worker_ids creates separate entries.
        """
        world_size = 2

        # Store same chunk hash but different worker_ids
        key_w0 = create_object_key(chunk_hash=100, worker_id=0, world_size=world_size)
        key_w1 = create_object_key(chunk_hash=100, worker_id=1, world_size=world_size)

        # Store worker 0's data
        reserved_dict0 = storage_manager.reserve_write([key_w0], test_layout)
        assert len(reserved_dict0) == 1
        storage_manager.finish_write(list(reserved_dict0.keys()))

        # Store worker 1's data
        reserved_dict1 = storage_manager.reserve_write([key_w1], test_layout)
        assert len(reserved_dict1) == 1
        storage_manager.finish_write(list(reserved_dict1.keys()))

        # Prefetch to secure both entries
        handle = storage_manager.submit_prefetch_task(
            single_row_spec([key_w0, key_w1], test_layout)
        )
        _ = storage_manager.query_prefetch_status(handle)

        # Both should be retrievable independently
        with storage_manager.read_prefetched_results([key_w0]) as objs:
            assert len(objs) == 1

        with storage_manager.read_prefetched_results([key_w1]) as objs:
            assert len(objs) == 1

    def test_tp2_retrieve_specific_worker(self, storage_manager, test_layout):
        """
        Test that retrieve with specific worker_id only gets that worker's data.
        """
        world_size = 2

        # Store for both workers
        all_keys = []
        for worker_id in range(world_size):
            keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(3)
            ]
            reserved_dict = storage_manager.reserve_write(keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))
            all_keys.extend(keys)

        # Prefetch to secure all entries
        handle = storage_manager.submit_prefetch_task(
            single_row_spec(all_keys, test_layout)
        )
        _ = storage_manager.query_prefetch_status(handle)

        # Retrieve only worker 0's data
        keys_w0 = [
            create_object_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(3)
        ]
        with storage_manager.read_prefetched_results(keys_w0) as objs:
            assert len(objs) == 3

        # Retrieve only worker 1's data
        keys_w1 = [
            create_object_key(chunk_hash=i, worker_id=1, world_size=world_size)
            for i in range(3)
        ]
        with storage_manager.read_prefetched_results(keys_w1) as objs:
            assert len(objs) == 3


# ==============================================================================
# Tests for Edge Cases
# ==============================================================================


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
class TestTPEdgeCases:
    """Edge case tests for tensor parallel support."""

    def test_world_size_1_stores_and_retrieves(self, storage_manager, test_layout):
        """
        Test that world_size=1 (no TP) works correctly through the API.
        Single worker stores and retrieves data successfully.
        """
        world_size = 1
        num_chunks = 3

        # Store chunks for worker 0
        storage_keys = [
            create_object_key(chunk_hash=i, worker_id=0, world_size=world_size)
            for i in range(num_chunks)
        ]
        reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Lookup should find all chunks
        handle = storage_manager.submit_prefetch_task(
            single_row_spec(storage_keys, test_layout)
        )
        found_count = storage_manager.query_prefetch_status(handle)[
            0
        ].count_leading_ones()
        assert found_count == num_chunks

        # Retrieve should work
        with storage_manager.read_prefetched_results(storage_keys) as objs:
            assert len(objs) == num_chunks

    def test_large_world_size_tp8(self, storage_manager, test_layout):
        """
        Test with larger world_size (TP=8) through the API.
        All 8 workers store and lookup works correctly.
        """
        world_size = 8
        num_chunks = 3

        # Store chunks for all workers
        all_keys = []
        for worker_id in range(world_size):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            all_keys.extend(storage_keys)
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Scheduler lookup: one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )
        assert hit_chunks == num_chunks
        for row in found_rows:
            assert row.get_indices_list() == list(range(num_chunks))

        # Verify retrieval for each worker
        for worker_id in range(world_size):
            worker_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            with storage_manager.read_prefetched_results(worker_keys) as objs:
                assert len(objs) == num_chunks

    def test_all_workers_same_chunk_different_keys(self, storage_manager, test_layout):
        """
        Test that same chunk_hash with different worker_ids creates
        distinct entries in storage and can be stored/retrieved independently.
        """
        world_size = 4
        chunk_hash = 42

        # Create storage keys for all workers with same chunk_hash
        storage_keys = [
            create_object_key(chunk_hash=chunk_hash, worker_id=i, world_size=world_size)
            for i in range(world_size)
        ]

        # All keys should be distinct
        assert len(set(storage_keys)) == world_size

        # Store all keys
        reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
        assert len(reserved_dict) == world_size
        storage_manager.finish_write(list(reserved_dict.keys()))

        # Lookup all keys
        handle = storage_manager.submit_prefetch_task(
            single_row_spec(storage_keys, test_layout)
        )
        found_count = storage_manager.query_prefetch_status(handle)[
            0
        ].count_leading_ones()
        assert found_count == world_size

        # Retrieve each worker's key independently
        for worker_id in range(world_size):
            worker_key = create_object_key(
                chunk_hash=chunk_hash, worker_id=worker_id, world_size=world_size
            )
            with storage_manager.read_prefetched_results([worker_key]) as objs:
                assert len(objs) == 1
                assert objs[0] is not None


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
class TestTPIntegration:
    """Integration tests simulating real TP workflows."""

    def test_full_tp2_workflow(self, storage_manager, test_layout):
        """
        Simulate a full TP=2 workflow:
        1. Worker 0 stores chunks 0, 1, 2
        2. Worker 1 stores chunks 0, 1, 2
        3. Scheduler looks up chunks 0, 1, 2, 3, 4 (one key row per worker)
        4. Verify correct hit count
        5. Workers retrieve their respective chunks
        """
        world_size = 2
        stored_chunks = 3
        requested_chunks = 5

        # Step 1 & 2: Workers store their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(stored_chunks)
            ]
            reserved_dict = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved_dict.keys()))

        # Step 3: Scheduler lookup, one key row per worker
        rows = create_lookup_key_rows(requested_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )

        # Step 4: Verify hit count -- both workers have chunks 0-2.
        assert hit_chunks == stored_chunks
        for row in found_rows:
            assert row.get_indices_list() == list(range(stored_chunks))

        # Step 5: Workers retrieve their chunks
        for worker_id in range(world_size):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(stored_chunks)
            ]
            with storage_manager.read_prefetched_results(storage_keys) as objs:
                assert len(objs) == stored_chunks
                for obj in objs:
                    assert obj is not None

    def test_concurrent_tp2_stores(self, storage_manager, test_layout):
        """
        Test concurrent stores from multiple "workers" (threads).
        """
        world_size = 2
        num_chunks = 10
        results = {}

        def worker_store(worker_id: int):
            storage_keys = [
                create_object_key(
                    chunk_hash=i, worker_id=worker_id, world_size=world_size
                )
                for i in range(num_chunks)
            ]
            reserved = storage_manager.reserve_write(storage_keys, test_layout)
            storage_manager.finish_write(list(reserved.keys()))
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

        # Verify lookup works with one key row per worker
        rows = create_lookup_key_rows(num_chunks, world_size)
        handle = storage_manager.submit_prefetch_task(ranked_spec(rows, test_layout))
        hit_chunks, found_rows = tp_lookup_hit_chunks(
            storage_manager, handle, world_size
        )
        assert hit_chunks == num_chunks
