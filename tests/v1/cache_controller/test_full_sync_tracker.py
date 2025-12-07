# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FullSyncTracker."""

# Standard
import time

# First Party
from lmcache.v1.cache_controller.controllers.full_sync_tracker import (
    FullSyncTracker,
)
from lmcache.v1.cache_controller.utils import (
    FullSyncState,
    RegistryTree,
    WorkerSyncInfo,
)


def create_tracker_with_registry() -> tuple[FullSyncTracker, RegistryTree]:
    """Helper to create a FullSyncTracker with its RegistryTree."""
    registry_tree = RegistryTree()
    tracker = FullSyncTracker(registry_tree=registry_tree)
    return tracker, registry_tree


def register_worker_in_registry(
    registry_tree: RegistryTree,
    instance_id: str,
    worker_id: int,
    ip: str = "192.168.1.1",
    port: int = 8000,
) -> None:
    """Helper to register a worker in RegistryTree."""
    registry_tree.register_worker(
        instance_id=instance_id,
        worker_id=worker_id,
        ip=ip,
        port=port,
        peer_init_url=None,
        socket=None,
        registration_time=time.time(),
    )


class TestWorkerSyncInfo:
    """Test cases for WorkerSyncInfo dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        info = WorkerSyncInfo(
            sync_id="test_sync_123",
            state=FullSyncState.SYNCING,
            start_time=1000.0,
            expected_total_keys=100,
            expected_batch_count=5,
        )

        assert info.sync_id == "test_sync_123"
        assert info.state == FullSyncState.SYNCING
        assert info.start_time == 1000.0
        assert info.expected_total_keys == 100
        assert info.expected_batch_count == 5
        assert info.received_batches == set()
        assert info.received_keys_count == 0
        assert info.last_activity_time == 1000.0  # auto-set from start_time

    def test_last_activity_time_auto_set(self):
        """Test that last_activity_time is auto-set from start_time."""
        info = WorkerSyncInfo(
            sync_id="test_sync",
            state=FullSyncState.SYNCING,
            start_time=2000.0,
            expected_total_keys=50,
            expected_batch_count=2,
        )
        assert info.last_activity_time == 2000.0


class TestFullSyncTracker:
    """Test cases for FullSyncTracker."""

    def test_init_defaults(self):
        """Test default initialization."""
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(registry_tree=registry_tree)

        assert tracker.completion_threshold == 0.8
        assert tracker.sync_timeout_s == 300.0
        assert tracker._need_full_sync_all is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(
            registry_tree=registry_tree,
            completion_threshold=0.9,
            sync_timeout_s=600.0,
        )

        assert tracker.completion_threshold == 0.9
        assert tracker.sync_timeout_s == 600.0

    def test_set_need_full_sync_all(self):
        """Test setting the global full sync flag."""
        tracker, _ = create_tracker_with_registry()

        assert tracker._need_full_sync_all is True
        tracker.set_need_full_sync_all(False)
        assert tracker._need_full_sync_all is False
        tracker.set_need_full_sync_all(True)
        assert tracker._need_full_sync_all is True

    def test_should_request_full_sync_controller_restart(self):
        """Test full sync request after controller restart."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)

        assert need_sync is True
        assert reason == "controller_restart"

    def test_should_request_full_sync_already_syncing(self):
        """Test that syncing worker is not requested to sync again."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Start sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)

        assert need_sync is False
        assert reason is None

    def test_should_request_full_sync_completed(self):
        """Test that completed worker is not requested to sync again."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Start and complete sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
        tracker.complete_sync("instance_1", 0, "sync_1", 100)

        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)

        assert need_sync is False
        assert reason is None

    def test_should_request_full_sync_failed(self):
        """Test that failed worker is requested to sync again."""
        tracker, registry_tree = create_tracker_with_registry()
        tracker.set_need_full_sync_all(False)  # Disable global flag to test retry logic
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Start and fail sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
        tracker.mark_failed("instance_1", 0, "timeout")

        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)

        assert need_sync is True
        assert reason == "sync_failed_retry"

    def test_is_worker_syncing(self):
        """Test checking if worker is syncing."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Before sync
        assert tracker.is_worker_syncing("instance_1", 0) is False

        # During sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
        assert tracker.is_worker_syncing("instance_1", 0) is True

        # After completion
        tracker.complete_sync("instance_1", 0, "sync_1", 100)
        assert tracker.is_worker_syncing("instance_1", 0) is False

    def test_start_sync_success(self):
        """Test successful sync start."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        result = tracker.start_sync("instance_1", 0, "sync_1", 1000, 10)

        assert result is True
        worker_node = registry_tree.get_worker("instance_1", 0)
        assert worker_node is not None
        sync_info = worker_node.sync_info
        assert sync_info is not None
        assert sync_info.sync_id == "sync_1"
        assert sync_info.state == FullSyncState.SYNCING
        assert sync_info.expected_total_keys == 1000
        assert sync_info.expected_batch_count == 10

    def test_start_sync_conflict(self):
        """Test sync start when another sync is in progress with different ID."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Start first sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        # Try to start another sync with different ID
        result = tracker.start_sync("instance_1", 0, "sync_2", 200, 10)

        assert result is False

    def test_start_sync_same_id(self):
        """Test sync start with same sync ID (retry scenario)."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)

        # Start first sync
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        # Start again with same ID (should succeed)
        result = tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        assert result is True

    def test_receive_batch_success(self):
        """Test successful batch receipt."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        result = tracker.receive_batch("instance_1", 0, "sync_1", 0, 20)

        assert result is True
        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert 0 in sync_info.received_batches
        assert sync_info.received_keys_count == 20

    def test_receive_batch_wrong_sync_id(self):
        """Test batch receipt with wrong sync ID."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        result = tracker.receive_batch("instance_1", 0, "wrong_sync_id", 0, 20)

        assert result is False

    def test_receive_batch_multiple(self):
        """Test receiving multiple batches."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        tracker.receive_batch("instance_1", 0, "sync_1", 0, 20)
        tracker.receive_batch("instance_1", 0, "sync_1", 1, 20)
        tracker.receive_batch("instance_1", 0, "sync_1", 2, 20)

        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.received_batches == {0, 1, 2}
        assert sync_info.received_keys_count == 60

    def test_complete_sync_success(self):
        """Test successful sync completion."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        # Receive all batches
        for i in range(5):
            tracker.receive_batch("instance_1", 0, "sync_1", i, 20)

        result = tracker.complete_sync("instance_1", 0, "sync_1", 100)

        assert result is True
        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.COMPLETED

    def test_complete_sync_wrong_sync_id(self):
        """Test sync completion with wrong sync ID."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        result = tracker.complete_sync("instance_1", 0, "wrong_sync_id", 100)

        assert result is False

    def test_mark_failed(self):
        """Test marking a sync as failed."""
        tracker, registry_tree = create_tracker_with_registry()
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        tracker.mark_failed("instance_1", 0, "test_reason")

        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.FAILED

    def test_check_sync_timeout(self):
        """Test sync timeout detection."""
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(
            registry_tree=registry_tree, sync_timeout_s=0.1
        )  # 100ms timeout
        register_worker_in_registry(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        # Wait for timeout
        time.sleep(0.2)

        tracker.check_sync_timeout()

        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.FAILED

    def test_get_global_progress(self):
        """Test global progress calculation."""
        tracker, registry_tree = create_tracker_with_registry()
        tracker.set_need_full_sync_all(False)  # Disable global flag for this test

        # Register 4 workers
        for i in range(4):
            register_worker_in_registry(registry_tree, "instance_1", i)

        assert tracker.get_global_progress() == 0.0

        # Complete 2 workers
        tracker.start_sync("instance_1", 0, "sync_0", 100, 5)
        tracker.complete_sync("instance_1", 0, "sync_0", 100)
        tracker.start_sync("instance_1", 1, "sync_1", 100, 5)
        tracker.complete_sync("instance_1", 1, "sync_1", 100)

        assert tracker.get_global_progress() == 0.5

    def test_can_exit_freeze(self):
        """Test freeze mode exit check."""
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(registry_tree=registry_tree, completion_threshold=0.5)

        # Register 4 workers
        for i in range(4):
            register_worker_in_registry(registry_tree, "instance_1", i)

        # Initially cannot exit
        assert tracker.can_exit_freeze() is False

        # Complete 2 workers (50%)
        tracker.start_sync("instance_1", 0, "sync_0", 100, 5)
        tracker.complete_sync("instance_1", 0, "sync_0", 100)
        tracker.start_sync("instance_1", 1, "sync_1", 100, 5)
        tracker.complete_sync("instance_1", 1, "sync_1", 100)

        # Now can exit (50% >= 50% threshold)
        assert tracker.can_exit_freeze() is True
        # Global flag should be disabled
        assert tracker._need_full_sync_all is False

    def test_get_sync_status(self):
        """Test getting sync status for a specific worker."""
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(registry_tree=registry_tree, completion_threshold=0.5)

        # Register 2 workers
        register_worker_in_registry(registry_tree, "instance_1", 0)
        register_worker_in_registry(registry_tree, "instance_1", 1)

        # Start sync for worker 0
        tracker.start_sync("instance_1", 0, "sync_0", 100, 5)

        is_complete, progress, can_exit = tracker.get_sync_status(
            "instance_1", 0, "sync_0"
        )
        assert is_complete is False
        assert progress == 0.0
        assert can_exit is False

        # Complete worker 0
        tracker.complete_sync("instance_1", 0, "sync_0", 100)

        is_complete, progress, can_exit = tracker.get_sync_status(
            "instance_1", 0, "sync_0"
        )
        assert is_complete is True
        assert progress == 0.5
        assert can_exit is True  # 50% >= 50% threshold

    def test_get_completed_count(self):
        """Test getting completed worker count."""
        tracker, registry_tree = create_tracker_with_registry()
        tracker.set_need_full_sync_all(False)

        for i in range(4):
            register_worker_in_registry(registry_tree, "instance_1", i)

        assert tracker.get_completed_count() == 0

        tracker.start_sync("instance_1", 0, "sync_0", 100, 5)
        tracker.complete_sync("instance_1", 0, "sync_0", 100)

        assert tracker.get_completed_count() == 1

    def test_get_syncing_count(self):
        """Test getting syncing worker count."""
        tracker, registry_tree = create_tracker_with_registry()
        tracker.set_need_full_sync_all(False)

        for i in range(4):
            register_worker_in_registry(registry_tree, "instance_1", i)

        assert tracker.get_syncing_count() == 0

        tracker.start_sync("instance_1", 0, "sync_0", 100, 5)
        tracker.start_sync("instance_1", 1, "sync_1", 100, 5)

        assert tracker.get_syncing_count() == 2

        tracker.complete_sync("instance_1", 0, "sync_0", 100)

        assert tracker.get_syncing_count() == 1
