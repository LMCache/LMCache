# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for per-user L2 eviction.

Tests the full interaction between MockL2Adapter (per-user byte tracking),
UserLRUEvictionPolicy, QuotaManager, and L2EvictionController working
together to enforce per-user storage quotas.
"""

# Standard
import os
import select

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# =============================================================================
# Helper Functions
# =============================================================================

# Size of each test object in bytes (256 float32 elements = 1024 bytes).
OBJ_NUM_ELEMENTS = 256


def make_key(
    chunk_id: int,
    user_id: str = "",
    model: str = "test",
    kv_rank: int = 0,
) -> ObjectKey:
    """Create an ObjectKey for testing.

    Args:
        chunk_id: Integer chunk hash.
        user_id: User identity.
        model: Model name.
        kv_rank: KV rank.

    Returns:
        ObjectKey with the given parameters.
    """
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model,
        kv_rank=kv_rank,
        user_id=user_id,
    )


def make_obj(
    num_elements: int = OBJ_NUM_ELEMENTS,
    fill: float = 1.0,
) -> TensorMemoryObj:
    """Create a TensorMemoryObj for testing.

    Args:
        num_elements: Number of float32 elements.
        fill: Fill value for the tensor.

    Returns:
        TensorMemoryObj wrapping a flat float32 tensor.
    """
    data = torch.full((num_elements,), fill, dtype=torch.float32)
    meta = MemoryObjMetadata(
        shape=torch.Size([num_elements]),
        dtype=torch.float32,
        address=0,
        phy_size=num_elements * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(data, meta, parent_allocator=None)


def wait_for_store(adapter: MockL2Adapter, timeout: float = 5.0) -> None:
    """Block until the adapter signals a store-complete event.

    Args:
        adapter: The mock adapter.
        timeout: Seconds to wait before raising.

    Raises:
        AssertionError: If the event fd is not signaled in time.
    """
    fd = adapter.get_store_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    assert events, "store event fd not signaled within timeout"
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass


def store_keys_for_user(
    adapter: MockL2Adapter,
    user_id: str,
    chunk_ids: list[int],
) -> list[ObjectKey]:
    """Store a batch of keys for a given user and wait for completion.

    Each key stores a 1024-byte (256 x float32) object.

    Args:
        adapter: The mock adapter to store into.
        user_id: User identity for the stored keys.
        chunk_ids: Integer chunk hashes identifying each key.

    Returns:
        The list of ObjectKeys that were stored.
    """
    keys = [make_key(cid, user_id=user_id) for cid in chunk_ids]
    objs: list[MemoryObj] = [make_obj() for _ in chunk_ids]
    adapter.submit_store_task(keys, objs)
    wait_for_store(adapter)
    adapter.pop_completed_store_tasks()
    return keys


def _obj_bytes() -> int:
    """Return the byte size of a single test object."""
    return OBJ_NUM_ELEMENTS * 4  # float32


def _make_adapter(
    capacity_bytes: int,
) -> MockL2Adapter:
    """Create a MockL2Adapter with the given byte capacity.

    Args:
        capacity_bytes: Maximum storage capacity in bytes.

    Returns:
        A MockL2Adapter instance.
    """
    gb = capacity_bytes / (1024**3)
    config = MockL2AdapterConfig(max_size_gb=gb, mock_bandwidth_gb=10.0)
    return MockL2Adapter(config)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTwoUsersOnlyViolatorEvicted:
    """Two users store keys; only the one exceeding quota is evicted."""

    def test_two_users_only_violator_evicted(self):
        """Alice exceeds her quota watermark, bob does not.

        After one eviction cycle only alice's keys should be partially
        evicted while bob's keys remain untouched.
        """
        obj_size = _obj_bytes()
        # Give adapter enough room for both users' keys.
        # Alice stores 5 keys, bob stores 3 keys => 8 * 1024 bytes.
        adapter = _make_adapter(capacity_bytes=obj_size * 20)
        try:
            eviction_config = EvictionConfig(
                eviction_policy="UserLRU",
                trigger_watermark=0.8,
                eviction_ratio=0.5,
            )
            quota_mgr = QuotaManager(total_capacity_bytes=obj_size * 20)

            # Alice quota: 3 objects worth of bytes.
            # She will store 5 objects => usage > 0.8 * quota.
            alice_quota_bytes = obj_size * 3
            quota_mgr.set_quota("alice", alice_quota_bytes / (1024**3))

            # Bob quota: 10 objects worth of bytes.
            # He will store 3 objects => usage well under watermark.
            bob_quota_bytes = obj_size * 10
            quota_mgr.set_quota("bob", bob_quota_bytes / (1024**3))

            state = L2AdapterEvictionState(adapter, eviction_config)
            controller = L2EvictionController([state], quota_manager=quota_mgr)

            alice_keys = store_keys_for_user(adapter, "alice", list(range(5)))
            bob_keys = store_keys_for_user(adapter, "bob", list(range(100, 103)))

            # Sanity: all keys present before eviction.
            for k in alice_keys + bob_keys:
                assert adapter.debug_has_key(k)

            # Run one eviction cycle.
            controller._check_and_evict(state)

            # Bob's keys must all survive.
            for k in bob_keys:
                assert adapter.debug_has_key(k), f"bob key {k} was unexpectedly evicted"

            # At least some of alice's keys must have been evicted.
            alice_remaining = sum(1 for k in alice_keys if adapter.debug_has_key(k))
            assert alice_remaining < len(alice_keys), (
                "expected some of alice's keys to be evicted"
            )
        finally:
            adapter.close()


class TestUnregisteredUserEvicted:
    """An unregistered user has effective limit=0; keys are evicted."""

    def test_unregistered_user_evicted(self):
        """Keys for a user with no quota entry should be evicted."""
        obj_size = _obj_bytes()
        adapter = _make_adapter(capacity_bytes=obj_size * 20)
        try:
            eviction_config = EvictionConfig(
                eviction_policy="UserLRU",
                trigger_watermark=0.8,
                eviction_ratio=1.0,
            )
            quota_mgr = QuotaManager(total_capacity_bytes=obj_size * 20)
            # No quota set for "ghost" — effective limit is 0.

            state = L2AdapterEvictionState(adapter, eviction_config)
            controller = L2EvictionController([state], quota_manager=quota_mgr)

            ghost_keys = store_keys_for_user(adapter, "ghost", list(range(4)))

            # Sanity: keys exist before eviction.
            for k in ghost_keys:
                assert adapter.debug_has_key(k)

            # One cycle with ratio=1.0 should evict everything.
            controller._check_and_evict(state)

            remaining = sum(1 for k in ghost_keys if adapter.debug_has_key(k))
            assert remaining == 0, (
                f"expected all ghost keys evicted, {remaining} remain"
            )
        finally:
            adapter.close()


class TestUserWithinQuotaUntouched:
    """A user whose usage is within quota watermark is not evicted."""

    def test_user_within_quota_untouched(self):
        """Store keys within quota, run eviction, verify nothing removed."""
        obj_size = _obj_bytes()
        adapter = _make_adapter(capacity_bytes=obj_size * 20)
        try:
            eviction_config = EvictionConfig(
                eviction_policy="UserLRU",
                trigger_watermark=0.8,
                eviction_ratio=0.5,
            )
            quota_mgr = QuotaManager(total_capacity_bytes=obj_size * 20)

            # Quota of 10 objects; store only 3 => 30% usage, below 80%.
            quota_mgr.set_quota("safe", obj_size * 10 / (1024**3))

            state = L2AdapterEvictionState(adapter, eviction_config)
            controller = L2EvictionController([state], quota_manager=quota_mgr)

            safe_keys = store_keys_for_user(adapter, "safe", list(range(3)))

            controller._check_and_evict(state)

            for k in safe_keys:
                assert adapter.debug_has_key(k), (
                    f"key {k} evicted despite being within quota"
                )
        finally:
            adapter.close()


class TestQuotaRemovedTriggersEviction:
    """Removing a user's quota at runtime causes their keys to be evicted."""

    def test_quota_removed_triggers_eviction(self):
        """Set quota, store within limit, remove quota, verify eviction."""
        obj_size = _obj_bytes()
        adapter = _make_adapter(capacity_bytes=obj_size * 20)
        try:
            eviction_config = EvictionConfig(
                eviction_policy="UserLRU",
                trigger_watermark=0.8,
                eviction_ratio=1.0,
            )
            quota_mgr = QuotaManager(total_capacity_bytes=obj_size * 20)

            # Initially within quota.
            quota_mgr.set_quota("temp", obj_size * 10 / (1024**3))

            state = L2AdapterEvictionState(adapter, eviction_config)
            controller = L2EvictionController([state], quota_manager=quota_mgr)

            temp_keys = store_keys_for_user(adapter, "temp", list(range(3)))

            # Confirm safe before removal.
            controller._check_and_evict(state)
            for k in temp_keys:
                assert adapter.debug_has_key(k)

            # Remove quota — effective limit drops to 0.
            quota_mgr.remove_quota("temp")

            controller._check_and_evict(state)

            remaining = sum(1 for k in temp_keys if adapter.debug_has_key(k))
            assert remaining == 0, (
                f"expected all keys evicted after quota removal, {remaining} remain"
            )
        finally:
            adapter.close()


class TestBackwardCompatLRUMode:
    """Regular LRU policy ignores per-user quotas and evicts globally."""

    def test_backward_compat_lru_mode(self):
        """Use LRU policy; verify global watermark eviction works."""
        obj_size = _obj_bytes()
        # Capacity for exactly 10 objects.
        capacity = obj_size * 10
        adapter = _make_adapter(capacity_bytes=capacity)
        try:
            eviction_config = EvictionConfig(
                eviction_policy="LRU",
                trigger_watermark=0.5,
                eviction_ratio=0.5,
            )
            # QuotaManager is provided but should be ignored for LRU.
            quota_mgr = QuotaManager(total_capacity_bytes=capacity)
            quota_mgr.set_quota("alice", 0.5 * capacity / (1024**3))

            state = L2AdapterEvictionState(adapter, eviction_config)
            controller = L2EvictionController([state], quota_manager=quota_mgr)

            # Store 8 objects (80% full) — above the 50% watermark.
            store_keys_for_user(adapter, "alice", list(range(8)))

            # Confirm all stored.
            assert adapter.debug_get_stored_object_count() == 8

            controller._check_and_evict(state)

            # Some keys should be evicted because global usage > watermark.
            remaining = adapter.debug_get_stored_object_count()
            assert remaining < 8, (
                f"expected LRU eviction to reduce count, got {remaining}"
            )
        finally:
            adapter.close()
