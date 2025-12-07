# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVController full sync handling."""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStatusMsg,
    KVAdmitMsg,
    KVEvictMsg,
)
from lmcache.v1.cache_controller.utils import FullSyncState, RegistryTree


@pytest.fixture
def kv_controller():
    """Create a KVController instance for testing."""
    registry = RegistryTree()
    controller = KVController(
        registry=registry,
        full_sync_completion_threshold=0.8,
        full_sync_timeout_s=300.0,
    )
    controller.cluster_executor = MagicMock()
    return controller


class TestKVControllerFullSyncStart:
    """Test cases for handle_full_sync_start."""

    @pytest.mark.asyncio
    async def test_full_sync_start_accepted(self, kv_controller):
        """Test successful full sync start."""
        # Register worker in both registry and tracker
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=1000,
            batch_count=10,
        )

        ret_msg = await kv_controller.handle_full_sync_start(msg)

        assert ret_msg.accepted is True
        assert ret_msg.sync_id == "sync_123"
        assert kv_controller.full_sync_tracker.is_worker_syncing("instance_1", 0)

    @pytest.mark.asyncio
    async def test_full_sync_start_clears_existing_keys(self, kv_controller):
        """Test that full sync start clears existing keys."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Pre-populate some keys
        for key in [1, 2, 3, 4, 5]:
            kv_controller.registry.admit_kv("instance_1", 0, "LocalCPUBackend", key)

        msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=100,
            batch_count=5,
        )

        await kv_controller.handle_full_sync_start(msg)

        # Keys should be cleared
        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert len(keys_in_pool) == 0

    @pytest.mark.asyncio
    async def test_full_sync_start_rejected_conflict(self, kv_controller):
        """Test full sync start rejected due to conflict."""
        # Register worker in both registry and tracker
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start first sync
        msg1 = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=1000,
            batch_count=10,
        )
        await kv_controller.handle_full_sync_start(msg1)

        # Try to start another with different sync_id
        msg2 = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_456",  # Different sync_id
            total_keys=2000,
            batch_count=20,
        )

        ret_msg = await kv_controller.handle_full_sync_start(msg2)

        assert ret_msg.accepted is False
        assert ret_msg.error_msg is not None


class TestKVControllerFullSyncBatch:
    """Test cases for handle_full_sync_batch."""

    @pytest.mark.asyncio
    async def test_full_sync_batch_adds_keys(self, kv_controller):
        """Test that batch adds keys to registry."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start sync first
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=100,
            batch_count=5,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Send batch
        batch_msg = FullSyncBatchMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            batch_id=0,
            keys=[1, 2, 3, 4, 5],
        )

        await kv_controller.handle_full_sync_batch(batch_msg)

        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert 1 in keys_in_pool
        assert 2 in keys_in_pool
        assert 5 in keys_in_pool
        assert len(keys_in_pool) == 5

    @pytest.mark.asyncio
    async def test_full_sync_multiple_batches(self, kv_controller):
        """Test multiple batch messages."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=15,
            batch_count=3,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Send 3 batches
        for batch_id in range(3):
            keys = list(range(batch_id * 5, (batch_id + 1) * 5))
            batch_msg = FullSyncBatchMsg(
                instance_id="instance_1",
                worker_id=0,
                location="LocalCPUBackend",
                sync_id="sync_123",
                batch_id=batch_id,
                keys=keys,
            )
            await kv_controller.handle_full_sync_batch(batch_msg)

        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert len(keys_in_pool) == 15
        for i in range(15):
            assert i in keys_in_pool


class TestKVControllerFullSyncEnd:
    """Test cases for handle_full_sync_end."""

    @pytest.mark.asyncio
    async def test_full_sync_end_completes_sync(self, kv_controller):
        """Test that sync end marks sync as completed."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=5,
            batch_count=1,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Send batch
        batch_msg = FullSyncBatchMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            batch_id=0,
            keys=[1, 2, 3, 4, 5],
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        # End sync
        end_msg = FullSyncEndMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            actual_total_keys=5,
        )

        await kv_controller.handle_full_sync_end(end_msg)

        # Sync should be completed
        assert not kv_controller.full_sync_tracker.is_worker_syncing("instance_1", 0)
        worker_node = kv_controller.registry.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.COMPLETED


class TestKVControllerFullSyncStatus:
    """Test cases for handle_full_sync_status."""

    @pytest.mark.asyncio
    async def test_full_sync_status_incomplete(self, kv_controller):
        """Test status query for incomplete sync."""
        # Register workers in both registry and tracker
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )
        kv_controller.registry.register_worker(
            "instance_1", 1, "127.0.0.1", 8001, None, MagicMock(), time.time()
        )

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=100,
            batch_count=5,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Query status
        status_msg = FullSyncStatusMsg(
            instance_id="instance_1",
            worker_id=0,
            sync_id="sync_123",
        )

        ret_msg = await kv_controller.handle_full_sync_status(status_msg)

        assert ret_msg.is_complete is False
        assert ret_msg.global_progress == 0.0
        assert ret_msg.can_exit_freeze is False

    @pytest.mark.asyncio
    async def test_full_sync_status_complete(self, kv_controller):
        """Test status query for complete sync."""
        # Import FullSyncTracker here to avoid circular dependency
        # First Party
        from lmcache.v1.cache_controller.controllers.full_sync_tracker import (
            FullSyncTracker,
        )

        # Use 50% threshold for easier testing
        kv_controller.full_sync_tracker = FullSyncTracker(
            registry_tree=kv_controller.registry,
            completion_threshold=0.5,
        )

        # Register workers in both registry and tracker
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )
        kv_controller.registry.register_worker(
            "instance_1", 1, "127.0.0.1", 8001, None, MagicMock(), time.time()
        )

        # Complete full sync for worker 0
        kv_controller.full_sync_tracker.start_sync("instance_1", 0, "sync_123", 100, 5)
        kv_controller.full_sync_tracker.complete_sync("instance_1", 0, "sync_123", 100)

        # Query status
        status_msg = FullSyncStatusMsg(
            instance_id="instance_1",
            worker_id=0,
            sync_id="sync_123",
        )

        ret_msg = await kv_controller.handle_full_sync_status(status_msg)

        assert ret_msg.is_complete is True
        assert ret_msg.global_progress == 0.5  # 1/2 workers
        assert ret_msg.can_exit_freeze is True  # >= 50%


class TestKVControllerIncrementalDiscardDuringSyc:
    """Test cases for incremental message discard during sync."""

    @pytest.mark.asyncio
    async def test_admit_discarded_during_sync(self, kv_controller):
        """Test that admit messages are discarded during sync."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=100,
            batch_count=5,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Try to admit a key (should be discarded)
        admit_msg = KVAdmitMsg(
            instance_id="instance_1",
            worker_id=0,
            key=999,
            location="LocalCPUBackend",
            seq_num=0,
        )
        await kv_controller.admit(admit_msg)

        # Key should NOT be in pool (discarded)
        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert 999 not in keys_in_pool

    @pytest.mark.asyncio
    async def test_evict_discarded_during_sync(self, kv_controller):
        """Test that evict messages are discarded during sync."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Pre-populate a key
        kv_controller.registry.admit_kv("instance_1", 0, "LocalCPUBackend", 100)

        # Start sync (which clears existing keys, but let's add one after start)
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="OtherBackend",  # Different location to not clear
            sync_id="sync_123",
            total_keys=100,
            batch_count=5,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Re-add the key via batch
        batch_msg = FullSyncBatchMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            batch_id=0,
            keys=[100],
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        # Try to evict (should be discarded)
        evict_msg = KVEvictMsg(
            instance_id="instance_1",
            worker_id=0,
            key=100,
            location="LocalCPUBackend",
            seq_num=0,
        )
        await kv_controller.evict(evict_msg)

        # Key should still be in pool (evict was discarded)
        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert 100 in keys_in_pool

    @pytest.mark.asyncio
    async def test_admit_allowed_after_sync_complete(self, kv_controller):
        """Test that admit works after sync is complete."""
        # Register worker in registry first
        kv_controller.registry.register_worker(
            "instance_1", 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )

        # Start and complete sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=0,
            batch_count=1,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        end_msg = FullSyncEndMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            actual_total_keys=0,
        )
        await kv_controller.handle_full_sync_end(end_msg)

        # Now admit should work
        admit_msg = KVAdmitMsg(
            instance_id="instance_1",
            worker_id=0,
            key=999,
            location="LocalCPUBackend",
            seq_num=0,
        )
        await kv_controller.admit(admit_msg)

        # Key should be in pool
        keys_in_pool = kv_controller.registry.get_worker_kv_keys(
            "instance_1", 0, "LocalCPUBackend"
        )
        assert 999 in keys_in_pool
