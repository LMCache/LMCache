# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterator
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_pending_store import (
    FIFOOffloadPolicy,
    LazyOffloadPendingStore,
    OffloadPolicy,
    PendingStoreItem,
)


def _make_item(request_id: str = "req-0") -> PendingStoreItem:
    """Helper to create a mock PendingStoreItem."""
    metadata = MagicMock()
    metadata.request_id = request_id
    return PendingStoreItem(metadata=metadata)


def _make_block_hashes(block_ids: list[int]) -> dict[int, bytes]:
    """Helper to create mock block hashes."""
    return {bid: f"hash-{bid}".encode() for bid in block_ids}


# ===========================================================================
# Tests for FIFOOffloadPolicy
# ===========================================================================


class TestFIFOOffloadPolicy:
    def test_init_default_threshold(self):
        policy = FIFOOffloadPolicy()
        assert policy.threshold == 100

    def test_init_custom_threshold(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 50}
        policy = FIFOOffloadPolicy(configs)
        assert policy.threshold == 50

    def test_add_item(self):
        policy = FIFOOffloadPolicy()
        item = _make_item()
        policy.add(item)
        assert len(policy.pending_items) == 1
        assert policy.pending_items[0] is item

    def test_should_offload_below_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        policy.add(_make_item("req-0"))
        policy.add(_make_item("req-1"))
        assert policy.should_offload() is False

    def test_should_offload_at_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        for i in range(3):
            policy.add(_make_item(f"req-{i}"))
        assert policy.should_offload() is True

    def test_should_offload_above_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_item(f"req-{i}"))
        assert policy.should_offload() is True

    def test_select_items_returns_iterator(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_item(f"req-{i}"))
        result = policy.select_items(3)
        assert isinstance(result, Iterator)

    def test_select_items_fifo_order(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        items = [_make_item(f"req-{i}") for i in range(5)]
        for item in items:
            policy.add(item)

        selected = list(policy.select_items(3))
        assert len(selected) == 3
        assert selected[0].metadata.request_id == "req-0"
        assert selected[1].metadata.request_id == "req-1"
        assert selected[2].metadata.request_id == "req-2"

    def test_select_items_removes_from_pending(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_item(f"req-{i}"))

        list(policy.select_items(3))
        assert len(policy.pending_items) == 2
        assert policy.pending_items[0].metadata.request_id == "req-3"
        assert policy.pending_items[1].metadata.request_id == "req-4"

    def test_select_items_count_exceeds_pending(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(3):
            policy.add(_make_item(f"req-{i}"))

        selected = list(policy.select_items(10))
        assert len(selected) == 3
        assert len(policy.pending_items) == 0

    def test_select_items_empty_queue(self):
        policy = FIFOOffloadPolicy()
        selected = list(policy.select_items(5))
        assert selected == []


# ===========================================================================
# Tests for LazyOffloadPendingStore
# ===========================================================================


class TestLazyOffloadPendingStore:
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

    def test_init_custom_select_count(self):
        configs = {"lmcache.mp.lazy_offload_select_count": 5}
        store = LazyOffloadPendingStore(configs)
        assert store._select_count == 5

    def test_add_stores_item_and_block_hashes(self):
        store = LazyOffloadPendingStore()
        item = _make_item("req-1")
        hashes = _make_block_hashes([10, 20, 30])
        store.add(item, hashes)

        assert store.get_block_hashes("req-1") == hashes

    def test_should_offload_delegates_to_policy(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 2}
        store = LazyOffloadPendingStore(configs)

        store.add(_make_item("req-0"), _make_block_hashes([0]))
        assert store.should_offload() is False

        store.add(_make_item("req-1"), _make_block_hashes([1]))
        assert store.should_offload() is True

    def test_select_items_returns_correct_count(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 2,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = LazyOffloadPendingStore(configs)

        for i in range(5):
            store.add(_make_item(f"req-{i}"), _make_block_hashes([i]))

        selected = list(store.select_items())
        assert len(selected) == 2

    def test_select_items_uses_select_count(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 3,
            "lmcache.mp.lazy_offload_select_count": 3,
        }
        store = LazyOffloadPendingStore(configs)

        for i in range(5):
            store.add(_make_item(f"req-{i}"), _make_block_hashes([i]))

        selected = list(store.select_items())
        assert len(selected) == 3
        assert selected[0].metadata.request_id == "req-0"
        assert selected[2].metadata.request_id == "req-2"

    def test_get_block_hashes_existing(self):
        store = LazyOffloadPendingStore()
        hashes = _make_block_hashes([1, 2, 3])
        store.add(_make_item("req-x"), hashes)
        assert store.get_block_hashes("req-x") == hashes

    def test_get_block_hashes_nonexistent(self):
        store = LazyOffloadPendingStore()
        assert store.get_block_hashes("no-such-req") == {}

    def test_remove_block_hashes(self):
        store = LazyOffloadPendingStore()
        store.add(_make_item("req-1"), _make_block_hashes([1]))
        store.remove_block_hashes("req-1")
        assert store.get_block_hashes("req-1") == {}

    def test_remove_block_hashes_nonexistent_no_error(self):
        store = LazyOffloadPendingStore()
        # Should not raise
        store.remove_block_hashes("nonexistent")

    def test_multiple_adds_same_request_id_overwrites_hashes(self):
        store = LazyOffloadPendingStore()
        hashes_1 = _make_block_hashes([1, 2])
        hashes_2 = _make_block_hashes([3, 4])
        store.add(_make_item("req-1"), hashes_1)
        store.add(_make_item("req-1"), hashes_2)
        # Latest hashes should overwrite
        assert store.get_block_hashes("req-1") == hashes_2

    def test_end_to_end_flow(self):
        """Test the full add -> should_offload -> select_items flow."""
        configs = {
            "lmcache.mp.lazy_offload_threshold": 3,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = LazyOffloadPendingStore(configs)

        # Add items below threshold
        for i in range(2):
            store.add(_make_item(f"req-{i}"), _make_block_hashes([i * 10]))
        assert store.should_offload() is False

        # Add one more to hit threshold
        store.add(_make_item("req-2"), _make_block_hashes([20]))
        assert store.should_offload() is True

        # Select items (select_count=2)
        selected = list(store.select_items())
        assert len(selected) == 2
        assert selected[0].metadata.request_id == "req-0"
        assert selected[1].metadata.request_id == "req-1"

        # After selection, below threshold again
        assert store.should_offload() is False

    def test_select_items_repeated_drains(self):
        """Verify multiple select_items calls drain the queue incrementally."""
        configs = {
            "lmcache.mp.lazy_offload_threshold": 2,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = LazyOffloadPendingStore(configs)

        for i in range(6):
            store.add(_make_item(f"req-{i}"), _make_block_hashes([i]))

        batch1 = list(store.select_items())
        assert len(batch1) == 2
        assert batch1[0].metadata.request_id == "req-0"

        batch2 = list(store.select_items())
        assert len(batch2) == 2
        assert batch2[0].metadata.request_id == "req-2"

        batch3 = list(store.select_items())
        assert len(batch3) == 2
        assert batch3[0].metadata.request_id == "req-4"


# ===========================================================================
# Tests for OffloadPolicy (abstract interface contract)
# ===========================================================================


class TestOffloadPolicyInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            OffloadPolicy()  # type: ignore[abstract]

    def test_custom_policy_integration(self):
        """Verify a custom policy can be plugged in via subclassing."""

        class LIFOOffloadPolicy(OffloadPolicy):
            """LIFO policy for testing extensibility."""

            def __init__(self, threshold: int = 3):
                self.pending_items: list[PendingStoreItem] = []
                self.threshold = threshold

            def add(self, item: PendingStoreItem):
                self.pending_items.append(item)

            def should_offload(self) -> bool:
                return len(self.pending_items) >= self.threshold

            def select_items(self, count: int) -> Iterator[PendingStoreItem]:
                # Return from the back (LIFO)
                to_offload = self.pending_items[-count:]
                self.pending_items = self.pending_items[:-count]
                return iter(reversed(to_offload))

        policy = LIFOOffloadPolicy(threshold=2)
        for i in range(4):
            policy.add(_make_item(f"req-{i}"))

        assert policy.should_offload() is True
        selected = list(policy.select_items(2))
        assert len(selected) == 2
        # LIFO: last added items first
        assert selected[0].metadata.request_id == "req-3"
        assert selected[1].metadata.request_id == "req-2"
