# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Full Sync message types."""

# Third Party
import msgspec

# First Party
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    Msg,
)


class TestHeartbeatMessages:
    """Test cases for Heartbeat message types."""

    def test_heartbeat_msg_creation(self):
        """Test HeartbeatMsg creation."""
        msg = HeartbeatMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url="tcp://192.168.1.1:9000",
        )

        assert msg.instance_id == "test_instance"
        assert msg.worker_id == 0
        assert msg.ip == "192.168.1.1"
        assert msg.port == 8000
        assert msg.peer_init_url == "tcp://192.168.1.1:9000"

    def test_heartbeat_msg_serialization(self):
        """Test HeartbeatMsg serialization/deserialization."""
        msg = HeartbeatMsg(
            instance_id="test_instance",
            worker_id=1,
            ip="192.168.1.2",
            port=8001,
            peer_init_url=None,
        )

        # Serialize
        encoded = msgspec.msgpack.encode(msg)
        # Deserialize
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HeartbeatMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 1
        assert decoded.peer_init_url is None

    def test_heartbeat_ret_msg_creation(self):
        """Test HeartbeatRetMsg creation."""
        msg = HeartbeatRetMsg(
            need_full_sync=True,
            full_sync_reason="controller_restart",
        )

        assert msg.need_full_sync is True
        assert msg.full_sync_reason == "controller_restart"

    def test_heartbeat_ret_msg_defaults(self):
        """Test HeartbeatRetMsg default values."""
        msg = HeartbeatRetMsg()

        assert msg.need_full_sync is False
        assert msg.full_sync_reason is None

    def test_heartbeat_ret_msg_serialization(self):
        """Test HeartbeatRetMsg serialization/deserialization."""
        msg = HeartbeatRetMsg(
            need_full_sync=True,
            full_sync_reason="sync_failed_retry",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HeartbeatRetMsg)
        assert decoded.need_full_sync is True
        assert decoded.full_sync_reason == "sync_failed_retry"


class TestFullSyncStartMessages:
    """Test cases for FullSyncStart message types."""

    def test_full_sync_start_msg_creation(self):
        """Test FullSyncStartMsg creation."""
        msg = FullSyncStartMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            total_keys=1000,
            batch_count=10,
        )

        assert msg.instance_id == "test_instance"
        assert msg.worker_id == 0
        assert msg.location == "LocalCPUBackend"
        assert msg.sync_id == "sync_12345"
        assert msg.total_keys == 1000
        assert msg.batch_count == 10

    def test_full_sync_start_msg_serialization(self):
        """Test FullSyncStartMsg serialization/deserialization."""
        msg = FullSyncStartMsg(
            instance_id="instance_1",
            worker_id=2,
            location="LocalCPUBackend",
            sync_id="sync_abc",
            total_keys=5000,
            batch_count=25,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStartMsg)
        assert decoded.sync_id == "sync_abc"
        assert decoded.total_keys == 5000

    def test_full_sync_start_ret_msg_creation(self):
        """Test FullSyncStartRetMsg creation."""
        msg = FullSyncStartRetMsg(
            sync_id="sync_12345",
            accepted=True,
            error_msg=None,
        )

        assert msg.sync_id == "sync_12345"
        assert msg.accepted is True
        assert msg.error_msg is None

    def test_full_sync_start_ret_msg_rejected(self):
        """Test FullSyncStartRetMsg for rejected case."""
        msg = FullSyncStartRetMsg(
            sync_id="sync_12345",
            accepted=False,
            error_msg="Worker already syncing",
        )

        assert msg.accepted is False
        assert msg.error_msg == "Worker already syncing"

    def test_full_sync_start_ret_msg_serialization(self):
        """Test FullSyncStartRetMsg serialization/deserialization."""
        msg = FullSyncStartRetMsg(
            sync_id="sync_xyz",
            accepted=True,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStartRetMsg)
        assert decoded.sync_id == "sync_xyz"
        assert decoded.accepted is True


class TestFullSyncBatchMessages:
    """Test cases for FullSyncBatch message type."""

    def test_full_sync_batch_msg_creation(self):
        """Test FullSyncBatchMsg creation."""
        keys = [1, 2, 3, 4, 5]
        msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            batch_id=0,
            keys=keys,
        )

        assert msg.instance_id == "test_instance"
        assert msg.batch_id == 0
        assert msg.keys == keys
        assert len(msg.keys) == 5

    def test_full_sync_batch_msg_large_batch(self):
        """Test FullSyncBatchMsg with large batch."""
        keys = list(range(2000))
        msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            batch_id=5,
            keys=keys,
        )

        assert len(msg.keys) == 2000
        assert msg.keys[0] == 0
        assert msg.keys[1999] == 1999

    def test_full_sync_batch_msg_serialization(self):
        """Test FullSyncBatchMsg serialization/deserialization."""
        keys = [100, 200, 300, 400, 500]
        msg = FullSyncBatchMsg(
            instance_id="instance_1",
            worker_id=1,
            location="LocalCPUBackend",
            sync_id="sync_batch_test",
            batch_id=3,
            keys=keys,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncBatchMsg)
        assert decoded.batch_id == 3
        assert decoded.keys == keys

    def test_full_sync_batch_msg_empty_keys(self):
        """Test FullSyncBatchMsg with empty keys."""
        msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            batch_id=0,
            keys=[],
        )

        assert len(msg.keys) == 0


class TestFullSyncEndMessages:
    """Test cases for FullSyncEnd message type."""

    def test_full_sync_end_msg_creation(self):
        """Test FullSyncEndMsg creation."""
        msg = FullSyncEndMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            actual_total_keys=1000,
        )

        assert msg.instance_id == "test_instance"
        assert msg.sync_id == "sync_12345"
        assert msg.actual_total_keys == 1000

    def test_full_sync_end_msg_serialization(self):
        """Test FullSyncEndMsg serialization/deserialization."""
        msg = FullSyncEndMsg(
            instance_id="instance_2",
            worker_id=3,
            location="LocalCPUBackend",
            sync_id="sync_end_test",
            actual_total_keys=99999,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncEndMsg)
        assert decoded.actual_total_keys == 99999


class TestFullSyncStatusMessages:
    """Test cases for FullSyncStatus message types."""

    def test_full_sync_status_msg_creation(self):
        """Test FullSyncStatusMsg creation."""
        msg = FullSyncStatusMsg(
            instance_id="test_instance",
            worker_id=0,
            sync_id="sync_12345",
        )

        assert msg.instance_id == "test_instance"
        assert msg.worker_id == 0
        assert msg.sync_id == "sync_12345"

    def test_full_sync_status_msg_serialization(self):
        """Test FullSyncStatusMsg serialization/deserialization."""
        msg = FullSyncStatusMsg(
            instance_id="instance_1",
            worker_id=2,
            sync_id="sync_status_test",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStatusMsg)
        assert decoded.sync_id == "sync_status_test"

    def test_full_sync_status_ret_msg_creation(self):
        """Test FullSyncStatusRetMsg creation."""
        msg = FullSyncStatusRetMsg(
            sync_id="sync_12345",
            is_complete=True,
            global_progress=0.85,
            can_exit_freeze=True,
        )

        assert msg.sync_id == "sync_12345"
        assert msg.is_complete is True
        assert msg.global_progress == 0.85
        assert msg.can_exit_freeze is True

    def test_full_sync_status_ret_msg_incomplete(self):
        """Test FullSyncStatusRetMsg for incomplete case."""
        msg = FullSyncStatusRetMsg(
            sync_id="sync_12345",
            is_complete=False,
            global_progress=0.3,
            can_exit_freeze=False,
        )

        assert msg.is_complete is False
        assert msg.global_progress == 0.3
        assert msg.can_exit_freeze is False

    def test_full_sync_status_ret_msg_serialization(self):
        """Test FullSyncStatusRetMsg serialization/deserialization."""
        msg = FullSyncStatusRetMsg(
            sync_id="sync_ret_test",
            is_complete=True,
            global_progress=1.0,
            can_exit_freeze=True,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStatusRetMsg)
        assert decoded.global_progress == 1.0
        assert decoded.can_exit_freeze is True


class TestMessageDescribe:
    """Test cases for message describe() methods."""

    def test_heartbeat_msg_describe(self):
        """Test HeartbeatMsg describe."""
        msg = HeartbeatMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
        )
        desc = msg.describe()
        assert "test_instance" in desc
        assert "0" in desc

    def test_heartbeat_ret_msg_describe(self):
        """Test HeartbeatRetMsg describe."""
        msg = HeartbeatRetMsg(need_full_sync=True, full_sync_reason="test_reason")
        desc = msg.describe()
        assert "True" in desc
        assert "test_reason" in desc

    def test_full_sync_start_msg_describe(self):
        """Test FullSyncStartMsg describe."""
        msg = FullSyncStartMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=1000,
            batch_count=10,
        )
        desc = msg.describe()
        assert "sync_123" in desc
        assert "1000" in desc

    def test_full_sync_batch_msg_describe(self):
        """Test FullSyncBatchMsg describe."""
        msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            batch_id=5,
            keys=[1, 2, 3],
        )
        desc = msg.describe()
        assert "sync_123" in desc
        assert "5" in desc

    def test_full_sync_end_msg_describe(self):
        """Test FullSyncEndMsg describe."""
        msg = FullSyncEndMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            actual_total_keys=1000,
        )
        desc = msg.describe()
        assert "sync_123" in desc
        assert "1000" in desc

    def test_full_sync_status_ret_msg_describe(self):
        """Test FullSyncStatusRetMsg describe."""
        msg = FullSyncStatusRetMsg(
            sync_id="sync_123",
            is_complete=True,
            global_progress=0.85,
            can_exit_freeze=True,
        )
        desc = msg.describe()
        assert "sync_123" in desc
        assert "True" in desc
