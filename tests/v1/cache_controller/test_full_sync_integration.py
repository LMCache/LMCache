# SPDX-License-Identifier: Apache-2.0
"""Integration tests for full sync functionality.

This module tests the complete full sync flow:
1. Controller detects need for full sync (e.g., after restart)
2. Worker receives need_full_sync flag in heartbeat response
3. Worker enters freeze mode and sends FullSyncStartMsg
4. Controller clears existing keys and marks worker as syncing
5. Worker sends keys in batches via FullSyncBatchMsg
6. Worker sends FullSyncEndMsg
7. Controller marks sync as complete
8. Worker queries status and exits freeze mode when threshold reached
"""

# Standard
from unittest.mock import MagicMock, patch
import asyncio

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.controllers.full_sync_tracker import (
    FullSyncTracker,
)
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.controllers.registration_controller import (
    RegistrationController,
)
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    KVAdmitMsg,
    Msg,
    RegisterMsg,
)
from lmcache.v1.cache_controller.utils import FullSyncState, RegistryTree


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, key_id, torch.bfloat16)


class MockZMQSocket:
    """Mock ZMQ socket for testing."""

    def __init__(self):
        self.sent_messages = []

    def send(self, data):
        self.sent_messages.append(data)


@pytest.fixture
def shared_registry():
    """Create a shared RegistryTree for testing."""
    return RegistryTree()


@pytest.fixture
def kv_controller(shared_registry):
    """Create a KVController for testing."""
    controller = KVController(
        registry=shared_registry,
        full_sync_completion_threshold=0.5,  # 50% for easier testing
        full_sync_timeout_s=300.0,
    )
    controller.cluster_executor = MagicMock()
    return controller


@pytest.fixture
def registration_controller(kv_controller, shared_registry):
    """Create a RegistrationController for testing."""
    controller = RegistrationController()
    # Replace the registry with the shared one
    controller.registry = shared_registry
    controller.kv_controller = kv_controller
    controller.cluster_executor = MagicMock()
    return controller


class TestFullSyncIntegrationFlow:
    """Integration tests for the complete full sync flow."""

    @pytest.mark.asyncio
    async def test_complete_sync_flow_single_worker(
        self, kv_controller, registration_controller
    ):
        """Test complete full sync flow for a single worker."""
        instance_id = "test_instance"
        worker_id = 0
        location = "LocalCPUBackend"
        sync_id = "sync_flow_test"

        # Step 1: Register worker
        register_msg = RegisterMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
        )
        # Mock socket creation
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_socket:
            mock_socket.return_value = MockZMQSocket()
            await registration_controller.register(register_msg)

        # Pre-populate some keys that should be cleared during sync
        for key in [100, 200, 300]:
            kv_controller.registry.admit_kv(instance_id, worker_id, location, key)

        # Step 2: Check heartbeat returns need_full_sync=True
        heartbeat_msg = HeartbeatMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
        )
        heartbeat_ret = await registration_controller.heartbeat(heartbeat_msg)

        assert heartbeat_ret.need_full_sync is True
        assert heartbeat_ret.full_sync_reason == "controller_restart"

        # Step 3: Start full sync
        start_msg = FullSyncStartMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id=sync_id,
            total_keys=10,
            batch_count=2,
        )
        start_ret = await kv_controller.handle_full_sync_start(start_msg)

        assert start_ret.accepted is True
        assert kv_controller.full_sync_tracker.is_worker_syncing(instance_id, worker_id)
        # Old keys should be cleared
        # Verify keys were cleared by checking registry
        assert (
            len(
                kv_controller.registry.get_worker_kv_keys(
                    instance_id, worker_id, location
                )
            )
            == 0
        )

        # Step 4: Send batches
        batch1_msg = FullSyncBatchMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id=sync_id,
            batch_id=0,
            keys=[1, 2, 3, 4, 5],
        )
        await kv_controller.handle_full_sync_batch(batch1_msg)

        batch2_msg = FullSyncBatchMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id=sync_id,
            batch_id=1,
            keys=[6, 7, 8, 9, 10],
        )
        await kv_controller.handle_full_sync_batch(batch2_msg)

        # Verify keys are added
        # All 10 keys should be in the registry now
        keys_in_registry = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert len(keys_in_registry) == 10
        assert keys_in_registry == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

        # Step 5: Send end message
        end_msg = FullSyncEndMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id=sync_id,
            actual_total_keys=10,
        )
        await kv_controller.handle_full_sync_end(end_msg)

        # Verify sync is completed
        assert not kv_controller.full_sync_tracker.is_worker_syncing(
            instance_id, worker_id
        )
        worker_node = kv_controller.registry.get_worker(instance_id, worker_id)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.COMPLETED

        # Step 6: Query status - should be able to exit freeze
        status_msg = FullSyncStatusMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            sync_id=sync_id,
        )
        status_ret = await kv_controller.handle_full_sync_status(status_msg)

        assert status_ret.is_complete is True
        assert status_ret.global_progress == 1.0  # Only 1 worker, 100%
        assert status_ret.can_exit_freeze is True

        # Step 7: Verify heartbeat no longer requests sync
        heartbeat_ret2 = await registration_controller.heartbeat(heartbeat_msg)
        assert heartbeat_ret2.need_full_sync is False

    @pytest.mark.asyncio
    async def test_sync_flow_multiple_workers(
        self, kv_controller, registration_controller
    ):
        """Test full sync flow with multiple workers."""
        location = "LocalCPUBackend"

        # Register 4 workers
        # Workers from the same instance use the same IP
        workers = [
            ("instance_1", 0, "192.168.1.1"),
            ("instance_1", 1, "192.168.1.1"),
            ("instance_2", 0, "192.168.1.20"),
            ("instance_2", 1, "192.168.1.20"),
        ]

        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_socket:
            mock_socket.return_value = MockZMQSocket()
            for instance_id, worker_id, ip in workers:
                register_msg = RegisterMsg(
                    instance_id=instance_id,
                    worker_id=worker_id,
                    ip=ip,
                    port=8000 + worker_id,
                    peer_init_url=None,
                )
                await registration_controller.register(register_msg)

        # Verify all workers need sync
        for instance_id, worker_id, ip in workers:
            heartbeat_msg = HeartbeatMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                ip=ip,
                port=8000 + worker_id,
                peer_init_url=None,
            )
            heartbeat_ret = await registration_controller.heartbeat(heartbeat_msg)
            assert heartbeat_ret.need_full_sync is True

        # Complete sync for first 2 workers (50%)
        for i, (instance_id, worker_id, _) in enumerate(workers[:2]):
            sync_id = f"sync_{instance_id}_{worker_id}"

            # Start
            start_msg = FullSyncStartMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                location=location,
                sync_id=sync_id,
                total_keys=5,
                batch_count=1,
            )
            await kv_controller.handle_full_sync_start(start_msg)

            # Batch
            batch_msg = FullSyncBatchMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                location=location,
                sync_id=sync_id,
                batch_id=0,
                keys=list(range(i * 5, (i + 1) * 5)),
            )
            await kv_controller.handle_full_sync_batch(batch_msg)

            # End
            end_msg = FullSyncEndMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                location=location,
                sync_id=sync_id,
                actual_total_keys=5,
            )
            await kv_controller.handle_full_sync_end(end_msg)

        # Check progress - should be 50% (2/4)
        assert kv_controller.full_sync_tracker.get_global_progress() == 0.5

        # With 50% threshold, freeze mode can be exited
        assert kv_controller.full_sync_tracker.can_exit_freeze() is True

        # Remaining workers should no longer need sync (global flag disabled)
        for instance_id, worker_id, _ in workers[2:]:
            need_sync, reason = (
                kv_controller.full_sync_tracker.should_request_full_sync(
                    instance_id, worker_id
                )
            )
            assert need_sync is False

    @pytest.mark.asyncio
    async def test_incremental_messages_discarded_during_sync(
        self, kv_controller, registration_controller
    ):
        """Test that incremental messages are discarded during sync."""
        instance_id = "test_instance"
        worker_id = 0
        location = "LocalCPUBackend"

        # Register worker
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_socket:
            mock_socket.return_value = MockZMQSocket()
            register_msg = RegisterMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                ip="192.168.1.1",
                port=8000,
                peer_init_url=None,
            )
            await registration_controller.register(register_msg)

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id="sync_test",
            total_keys=5,
            batch_count=1,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Record keys before incremental admit
        keys_before = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        ).copy()

        # Send incremental admit - should be discarded
        admit_msg = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=999,
            location=location,
            seq_num=0,
        )
        await kv_controller.admit(admit_msg)

        # Verify key was NOT added
        # The incremental admit should be discarded during sync
        keys_after = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert keys_after == keys_before
        assert 999 not in keys_after

        # Complete sync
        batch_msg = FullSyncBatchMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id="sync_test",
            batch_id=0,
            keys=[1, 2, 3, 4, 5],
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        end_msg = FullSyncEndMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location=location,
            sync_id="sync_test",
            actual_total_keys=5,
        )
        await kv_controller.handle_full_sync_end(end_msg)

        # Now incremental admit should work
        admit_msg2 = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=1000,
            location=location,
            seq_num=1,
        )
        await kv_controller.admit(admit_msg2)

        # Verify key was added
        keys_final = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 1000 in keys_final


class TestFullSyncErrorHandling:
    """Tests for error handling in full sync."""

    @pytest.mark.asyncio
    async def test_sync_start_conflict(self, kv_controller, shared_registry):
        """Test handling of sync start conflict."""
        # Standard
        import time

        # Register worker in both RegistryTree and FullSyncTracker
        shared_registry.register_worker(
            instance_id="instance_1",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        # Start first sync
        start_msg1 = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_1",
            total_keys=100,
            batch_count=5,
        )
        ret1 = await kv_controller.handle_full_sync_start(start_msg1)
        assert ret1.accepted is True

        # Try to start another sync with different ID
        start_msg2 = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_2",
            total_keys=200,
            batch_count=10,
        )
        ret2 = await kv_controller.handle_full_sync_start(start_msg2)
        assert ret2.accepted is False
        assert ret2.error_msg is not None

    @pytest.mark.asyncio
    async def test_sync_timeout_marks_failed(self, kv_controller, shared_registry):
        """Test that sync timeout marks worker as failed."""
        # Standard
        import time as time_module

        # Use very short timeout for testing
        kv_controller.full_sync_tracker = FullSyncTracker(
            registry_tree=kv_controller.registry,
            completion_threshold=0.8,
            sync_timeout_s=0.1,  # 100ms
        )
        # Disable the global flag to test retry logic
        kv_controller.full_sync_tracker.set_need_full_sync_all(False)
        # Register worker in both RegistryTree and FullSyncTracker
        shared_registry.register_worker(
            instance_id="instance_1",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time_module.time(),
        )

        # Start sync but don't complete it
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_timeout_test",
            total_keys=100,
            batch_count=5,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Check timeout
        kv_controller.full_sync_tracker.check_sync_timeout()

        # Worker should be marked as failed
        worker_node = kv_controller.registry.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.FAILED

        # Should need re-sync
        need_sync, reason = kv_controller.full_sync_tracker.should_request_full_sync(
            "instance_1", 0
        )
        assert need_sync is True
        assert reason == "sync_failed_retry"

    @pytest.mark.asyncio
    async def test_batch_with_wrong_sync_id(self, kv_controller, shared_registry):
        """Test handling of batch with wrong sync ID."""
        # Standard
        import time

        location = "LocalCPUBackend"
        # Register worker in both RegistryTree and FullSyncTracker
        shared_registry.register_worker(
            instance_id="instance_1",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        # Start sync
        start_msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=0,
            location=location,
            sync_id="sync_correct",
            total_keys=10,
            batch_count=2,
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Record keys before batch
        keys_before = shared_registry.get_worker_kv_keys(
            "instance_1", 0, location
        ).copy()

        # Send batch with wrong sync ID
        batch_msg = FullSyncBatchMsg(
            instance_id="instance_1",
            worker_id=0,
            location=location,
            sync_id="sync_wrong",  # Wrong ID
            batch_id=0,
            keys=[1, 2, 3, 4, 5],
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        # Keys should NOT be added (batch was rejected)
        keys_after = shared_registry.get_worker_kv_keys("instance_1", 0, location)
        assert keys_after == keys_before


class TestMessageSerialization:
    """Tests for message serialization in full sync flow."""

    def test_full_sync_message_roundtrip(self):
        """Test that all full sync messages can be serialized and deserialized."""
        messages = [
            HeartbeatMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="192.168.1.1",
                port=8000,
                peer_init_url=None,
            ),
            HeartbeatRetMsg(
                need_full_sync=True,
                full_sync_reason="controller_restart",
            ),
            FullSyncStartMsg(
                instance_id="test_instance",
                worker_id=0,
                location="LocalCPUBackend",
                sync_id="sync_123",
                total_keys=1000,
                batch_count=10,
            ),
            FullSyncStartRetMsg(
                sync_id="sync_123",
                accepted=True,
            ),
            FullSyncBatchMsg(
                instance_id="test_instance",
                worker_id=0,
                location="LocalCPUBackend",
                sync_id="sync_123",
                batch_id=0,
                keys=[1, 2, 3, 4, 5],
            ),
            FullSyncEndMsg(
                instance_id="test_instance",
                worker_id=0,
                location="LocalCPUBackend",
                sync_id="sync_123",
                actual_total_keys=1000,
            ),
            FullSyncStatusMsg(
                instance_id="test_instance",
                worker_id=0,
                sync_id="sync_123",
            ),
            FullSyncStatusRetMsg(
                sync_id="sync_123",
                is_complete=True,
                global_progress=0.85,
                can_exit_freeze=True,
            ),
        ]

        for original_msg in messages:
            # Serialize
            encoded = msgspec.msgpack.encode(original_msg)
            # Deserialize
            decoded = msgspec.msgpack.decode(encoded, type=Msg)

            # Verify type
            assert type(decoded) is type(original_msg)

            # Verify key fields
            if hasattr(original_msg, "sync_id"):
                assert decoded.sync_id == original_msg.sync_id
            if hasattr(original_msg, "instance_id"):
                assert decoded.instance_id == original_msg.instance_id

    def test_large_batch_serialization(self):
        """Test serialization of large batch messages."""
        # Create a batch with many keys (simulating real scenario)
        keys = list(range(100000))  # 100K keys

        msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_large",
            batch_id=0,
            keys=keys,
        )

        # Serialize
        encoded = msgspec.msgpack.encode(msg)
        # Deserialize
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncBatchMsg)
        assert len(decoded.keys) == 100000
        assert decoded.keys == keys
