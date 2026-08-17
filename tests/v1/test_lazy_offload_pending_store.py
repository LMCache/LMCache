# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_pending_store import LazyOffloadPendingStore
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy


def _make_meta(request_id: str = "req-0", num_blocks: int = 1) -> MagicMock:
    """Helper to create a mock LMCacheMPRequestMetadata."""
    meta = MagicMock()
    meta.request_id = request_id
    meta.op.flat_block_ids = list(range(num_blocks))
    return meta


def _make_block_hashes(block_ids: list[int]) -> dict[int, bytes]:
    """Helper to create mock block hashes."""
    return {bid: f"hash-{bid}".encode() for bid in block_ids}


# ===========================================================================
# Tests for FIFOOffloadPolicy
# ===========================================================================


class TestFIFOOffloadPolicy:
    def test_init_default_threshold(self):
        policy = FIFOOffloadPolicy()
        assert policy._threshold == 100

    def test_init_custom_threshold(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 50}
        policy = FIFOOffloadPolicy(configs)
        assert policy._threshold == 50

    def test_add_creates_new_item(self):
        policy = FIFOOffloadPolicy()
        meta = _make_meta("req-0")
        hashes = _make_block_hashes([0, 1])
        policy.add(meta, hashes)
        assert "req-0" in policy._pending_items
        assert len(policy._pending_items["req-0"].metadatas) == 1

    def test_add_same_request_appends_metadatas(self):
        policy = FIFOOffloadPolicy()
        meta1 = _make_meta("req-0", num_blocks=1)
        meta2 = _make_meta("req-0", num_blocks=2)
        policy.add(meta1, _make_block_hashes([0]))
        policy.add(meta2, _make_block_hashes([0, 1]))
        assert len(policy._pending_items["req-0"].metadatas) == 2

    def test_pop_items_for_offload_below_threshold_returns_empty(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        assert policy.pop_items_for_offload(10) == []
        assert "req-0" in policy._pending_items

    def test_pop_items_for_offload_at_threshold_returns_finished_items(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        for i in range(3):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert len(policy.pop_items_for_offload(10)) == 3

    def test_mark_req_finished_not_in_pending_raises(self):
        policy = FIFOOffloadPolicy()
        with pytest.raises(ValueError, match="not in pending_items"):
            policy.mark_req_finished("nonexistent")

    def test_pop_items_for_offload_returns_only_finished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        # req-1 is not finished

        selected = policy.pop_items_for_offload(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-0"

    def test_pop_items_for_offload_removes_from_pending(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.pop_items_for_offload(10)
        assert len(selected) == 2
        assert len(policy._pending_items) == 0
        assert policy._finished_requests_count == 0

    def test_pop_items_for_offload_skips_unfinished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.pop_items_for_offload(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-1"
        # req-0 still pending
        assert "req-0" in policy._pending_items

    def test_pop_items_for_offload_count_limits_output(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")

        selected = policy.pop_items_for_offload(2)
        assert len(selected) == 2
        assert len(policy._pending_items) == 3

    def test_pop_items_for_offload_empty(self):
        policy = FIFOOffloadPolicy()
        assert policy.pop_items_for_offload(5) == []


# ===========================================================================
# Tests for LazyOffloadPendingStore
# ===========================================================================


class TestLazyOffloadPendingStore:
    def _setup_store_with_gpu_pool(self, configs=None):
        store = LazyOffloadPendingStore(configs)
        gpu_pool = MagicMock()
        gpu_pool.blocks = {
            bid: MagicMock(block_hash=f"hash-{bid}".encode()) for bid in range(10)
        }
        store.bind_gpu_block_pool(gpu_pool)
        return store

    def test_init_default_policy(self):
        store = LazyOffloadPendingStore()
        assert isinstance(store._policy, FIFOOffloadPolicy)

    def test_init_fifo_policy_explicit(self):
        configs = {"lmcache.mp.lazy_offload_policy": "FIFO"}
        store = LazyOffloadPendingStore(configs)
        assert isinstance(store._policy, FIFOOffloadPolicy)

    def test_init_unknown_policy_raises(self):
        configs = {"lmcache.mp.lazy_offload_policy": "UNKNOWN"}
        with pytest.raises(ValueError, match="Unknown offload policy"):
            LazyOffloadPendingStore(configs)

    def test_init_default_select_count(self):
        store = LazyOffloadPendingStore()
        assert store._select_count == 10

    def test_init_custom_select_count(self):
        configs = {"lmcache.mp.lazy_offload_select_count": 5}
        store = LazyOffloadPendingStore(configs)
        assert store._select_count == 5

    def test_bind_gpu_block_pool(self):
        store = LazyOffloadPendingStore()
        gpu_pool = MagicMock()
        store.bind_gpu_block_pool(gpu_pool)
        assert store._gpu_block_pool is gpu_pool

    def test_add_without_gpu_pool_raises(self):
        store = LazyOffloadPendingStore()
        meta = _make_meta("req-0")
        with pytest.raises(ValueError, match="gpu block pool not bound"):
            store.add(meta)

    def test_add_with_gpu_pool(self):
        store = self._setup_store_with_gpu_pool()
        meta = _make_meta("req-0", num_blocks=2)
        store.add(meta)
        # Verify block hashes were computed from gpu pool
        pending = store._policy._pending_items["req-0"]
        assert len(pending.metadatas) == 1
        assert pending.metadatas[0][1] == {0: b"hash-0", 1: b"hash-1"}

    def test_pop_items_for_offload_returns_empty_until_threshold(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 2}
        store = self._setup_store_with_gpu_pool(configs)

        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.pop_items_for_offload() == []

        store.add(_make_meta("req-1"))
        store.mark_req_finished("req-1")
        assert len(store.pop_items_for_offload()) == 2

    def test_pop_items_for_offload_returns_correct_count(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 3,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(5):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        selected = store.pop_items_for_offload()
        assert len(selected) == 3

    def test_mark_req_finished(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 1}
        store = self._setup_store_with_gpu_pool(configs)
        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert [item.request_id for item in store.pop_items_for_offload()] == ["req-0"]

    def test_update_get_remove_gpu_block_ids(self):
        store = LazyOffloadPendingStore()
        store.update_request_gpu_block_ids("req-0", [1, 2])
        store.update_request_gpu_block_ids("req-0", [3])
        assert store.get_request_gpu_block_ids("req-0") == [1, 2, 3]

        store.remove_request_gpu_block_ids("req-0")
        assert store.get_request_gpu_block_ids("req-0") == []

    def test_get_gpu_block_ids_nonexistent_returns_empty(self):
        store = LazyOffloadPendingStore()
        assert store.get_request_gpu_block_ids("nonexistent") == []

    def test_end_to_end_flow(self):
        """Test full add -> mark_finished -> pop_items_for_offload flow."""
        configs = {
            "lmcache.mp.lazy_offload_threshold": 3,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = self._setup_store_with_gpu_pool(configs)

        # Add items and mark them finished
        for i in range(5):
            store.add(_make_meta(f"req-{i}", num_blocks=1))
        for i in range(5):
            store.mark_req_finished(f"req-{i}")

        # Select first 2 (select_count=2)
        selected = store.pop_items_for_offload()
        assert len(selected) == 2
        assert selected[0].request_id == "req-0"
        assert selected[1].request_id == "req-1"

    def test_pop_items_for_offload_multiple_batches(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(6):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        batch1 = store.pop_items_for_offload()
        assert len(batch1) == 2
        assert batch1[0].request_id == "req-0"

        batch2 = store.pop_items_for_offload()
        assert len(batch2) == 2
        assert batch2[0].request_id == "req-2"

        batch3 = store.pop_items_for_offload()
        assert len(batch3) == 2
        assert batch3[0].request_id == "req-4"

        batch4 = store.pop_items_for_offload()
        assert len(batch4) == 0
