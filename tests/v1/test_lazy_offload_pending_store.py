# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import sys

# Mock vllm related imports
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.distributed", MagicMock())
sys.modules.setdefault("vllm.distributed.kv_transfer", MagicMock())
sys.modules.setdefault("vllm.distributed.kv_transfer.kv_connector", MagicMock())
sys.modules.setdefault("vllm.distributed.kv_transfer.kv_connector.v1", MagicMock())
sys.modules.setdefault("vllm.distributed.kv_transfer.kv_connector.v1.base", MagicMock())
sys.modules.setdefault("vllm.v1", MagicMock())
sys.modules.setdefault("vllm.v1.utils", MagicMock())

# Third Party
import pytest  # noqa: E402

# First Party
from lmcache.integration.vllm.lazy_offload_pending_store import (  # noqa: E402
    FIFOOffloadPolicy,
    LazyOffloadPendingStore,
    OffloadPolicy,
    PendingStoreItem,
)


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

    def test_should_offload_below_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        assert policy._finished_requests_count == 1
        assert policy.should_offload() is False

    def test_should_offload_at_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        for i in range(3):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_should_offload_above_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_mark_req_finished_not_in_pending_raises(self):
        policy = FIFOOffloadPolicy()
        with pytest.raises(ValueError, match="not in pending_items"):
            policy.mark_req_finished("nonexistent")

    def test_select_items_returns_only_finished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        # req-1 is not finished

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-0"

    def test_select_items_removes_from_pending(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 2
        assert len(policy._pending_items) == 0
        assert policy._finished_requests_count == 0

    def test_select_items_skips_unfinished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-1"
        # req-0 still pending
        assert "req-0" in policy._pending_items

    def test_select_items_count_limits_output(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")

        selected = policy.select_items(2)
        assert len(selected) == 2
        assert len(policy._pending_items) == 3

    def test_select_items_empty(self):
        policy = FIFOOffloadPolicy()
        assert policy.select_items(5) == []


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

    def test_should_offload_delegates_to_policy(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 2}
        store = self._setup_store_with_gpu_pool(configs)

        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is False

        store.add(_make_meta("req-1"))
        store.mark_req_finished("req-1")
        assert store.should_offload() is True

    def test_select_items_returns_correct_count(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 3,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(5):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        selected = store.select_items()
        assert len(selected) == 3

    def test_mark_req_finished(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 1}
        store = self._setup_store_with_gpu_pool(configs)
        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is True

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
        """Test full add -> mark_finished -> should_offload -> select_items."""
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

        # Should be over threshold
        assert store.should_offload() is True

        # Select first 2 (select_count=2)
        selected = store.select_items()
        assert len(selected) == 2
        assert selected[0].request_id == "req-0"
        assert selected[1].request_id == "req-1"

    def test_select_items_multiple_batches(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(6):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        batch1 = store.select_items()
        assert len(batch1) == 2
        assert batch1[0].request_id == "req-0"

        batch2 = store.select_items()
        assert len(batch2) == 2
        assert batch2[0].request_id == "req-2"

        batch3 = store.select_items()
        assert len(batch3) == 2
        assert batch3[0].request_id == "req-4"

        batch4 = store.select_items()
        assert len(batch4) == 0


# ===========================================================================
# Tests for OffloadPolicy (abstract interface contract)
# ===========================================================================


class TestOffloadPolicyInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            OffloadPolicy()  # type: ignore[abstract]

    def test_custom_policy_integration(self):
        """Verify a custom policy can be plugged in via subclassing."""

        class CustomOffloadPolicy(OffloadPolicy):
            def __init__(self):
                self.pending_items: dict[str, PendingStoreItem] = {}
                self.finished_count = 0
                self.threshold = 2

            def add(self, meta, block_hashes):
                if meta.request_id not in self.pending_items:
                    self.pending_items[meta.request_id] = PendingStoreItem(
                        request_id=meta.request_id
                    )
                self.pending_items[meta.request_id].metadatas.append(
                    (meta, block_hashes)
                )

            def mark_req_finished(self, req_id: str):
                self.pending_items[req_id].is_finished = True
                self.finished_count += 1

            def should_offload(self) -> bool:
                return self.finished_count >= self.threshold

            def select_items(self, count: int) -> list[PendingStoreItem]:
                result = []
                for req_id in self.pending_items:
                    if self.pending_items[req_id].is_finished:
                        result.append(self.pending_items[req_id])
                        del self.pending_items[req_id]
                return result

        policy = CustomOffloadPolicy()
        store = LazyOffloadPendingStore()
        store._policy = policy  # inject custom policy

        gpu_pool = MagicMock()
        gpu_pool.blocks = {
            bid: MagicMock(block_hash=f"hash-{bid}".encode()) for bid in range(5)
        }
        store.bind_gpu_block_pool(gpu_pool)

        store.add(_make_meta("req-0"))
        store.add(_make_meta("req-1"))
        store.mark_req_finished("req-0")
        store.mark_req_finished("req-1")

        assert store.should_offload() is True
        selected = store.select_items()
        assert len(selected) == 2
