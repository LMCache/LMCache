# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FullSyncSender."""
# Standard

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.full_sync_sender import FullSyncSender


def create_test_config(
    batch_size: int = 100,
    batch_interval_ms: int = 0,
    startup_delay_s: float = 0.0,
    status_poll_interval_s: float = 0.01,
    max_retry_count: int = 3,
    retry_delay_s: float = 0.01,
):
    """Create a test configuration."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_instance",
    )
    config.extra_config = {
        "full_sync_batch_size": batch_size,
        "full_sync_batch_interval_ms": batch_interval_ms,
        "full_sync_startup_delay_s": startup_delay_s,
        "full_sync_status_poll_interval_s": status_poll_interval_s,
        "full_sync_max_retry_count": max_retry_count,
        "full_sync_retry_delay_s": retry_delay_s,
    }
    return config


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, key_id, torch.bfloat16)


class MockWorker:
    """Mock LMCacheWorker for testing."""

    def __init__(self):
        self.worker_id = 0
        self.messages = []
        self.req_responses = []
        self._response_index = 0

    def put_msg(self, msg):
        """Record pushed messages."""
        self.messages.append(msg)

    async def async_put_and_wait_msg(self, msg):
        """Return predefined responses for REQ-REP messages."""
        if self._response_index < len(self.req_responses):
            response = self.req_responses[self._response_index]
            self._response_index += 1
            return response
        # Default responses
        if isinstance(msg, FullSyncStartMsg):
            return FullSyncStartRetMsg(sync_id=msg.sync_id, accepted=True)
        elif isinstance(msg, FullSyncStatusMsg):
            return FullSyncStatusRetMsg(
                sync_id=msg.sync_id,
                is_complete=True,
                global_progress=1.0,
                can_exit_freeze=True,
            )
        return None

    def set_responses(self, responses):
        """Set predefined responses."""
        self.req_responses = responses
        self._response_index = 0


class MockLMCacheEngine:
    """Mock LMCacheEngine for testing."""

    def __init__(self):
        self.freeze_mode = False

    def set_hot_cache_freeze_mode(self, mode: bool):
        """Record freeze mode changes."""
        self.freeze_mode = mode


class MockLocalCPUBackend:
    """Mock LocalCPUBackend for testing."""

    def __init__(self, keys=None):
        self._keys = keys or []

    def get_keys(self):
        """Return predefined keys."""
        return self._keys

    def __str__(self):
        return "LocalCPUBackend"


@pytest.fixture
def full_sync_sender():
    """Create a FullSyncSender instance for testing."""
    config = create_test_config()
    worker = MockWorker()
    engine = MockLMCacheEngine()
    backend = MockLocalCPUBackend()

    sender = FullSyncSender(
        config=config,
        worker=worker,
        lmcache_engine=engine,
        local_cpu_backend=backend,
    )
    return sender, worker, engine, backend


class TestFullSyncSenderInit:
    """Test cases for FullSyncSender initialization."""

    def test_init_default_config(self, full_sync_sender):
        """Test initialization with default config."""
        sender, _, _, _ = full_sync_sender

        assert sender.batch_size == 100
        assert sender.batch_interval_ms == 0
        assert sender.startup_delay_range_s == 0.0
        assert sender.max_retry_count == 3
        assert sender._is_syncing is False
        assert sender._current_sync_id is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = create_test_config(
            batch_size=500,
            batch_interval_ms=10,
            startup_delay_s=2.0,
            max_retry_count=5,
        )
        worker = MockWorker()
        engine = MockLMCacheEngine()
        backend = MockLocalCPUBackend()

        sender = FullSyncSender(
            config=config,
            worker=worker,
            lmcache_engine=engine,
            local_cpu_backend=backend,
        )

        assert sender.batch_size == 500
        assert sender.batch_interval_ms == 10
        assert sender.startup_delay_range_s == 2.0
        assert sender.max_retry_count == 5


class TestFullSyncSenderProperties:
    """Test cases for FullSyncSender properties."""

    def test_instance_id(self, full_sync_sender):
        """Test instance_id property."""
        sender, _, _, _ = full_sync_sender
        assert sender.instance_id == "test_instance"

    def test_worker_id(self, full_sync_sender):
        """Test worker_id property."""
        sender, _, _, _ = full_sync_sender
        assert sender.worker_id == 0

    def test_location(self, full_sync_sender):
        """Test location property."""
        sender, _, _, _ = full_sync_sender
        assert sender.location == "LocalCPUBackend"

    def test_is_syncing(self, full_sync_sender):
        """Test is_syncing property."""
        sender, _, _, _ = full_sync_sender
        assert sender.is_syncing is False


class TestFullSyncSenderSyncId:
    """Test cases for sync ID generation."""

    def test_generate_sync_id(self, full_sync_sender):
        """Test sync ID generation."""
        sender, _, _, _ = full_sync_sender

        sync_id = sender._generate_sync_id()

        assert sync_id.startswith("test_instance_0_")
        assert len(sync_id) > len("test_instance_0_")

    def test_generate_sync_id_unique(self, full_sync_sender):
        """Test that sync IDs are unique."""
        sender, _, _, _ = full_sync_sender

        sync_ids = [sender._generate_sync_id() for _ in range(100)]

        assert len(set(sync_ids)) == 100  # All unique


class TestFullSyncSenderGetKeys:
    """Test cases for getting hot cache keys."""

    def test_get_all_hot_cache_keys_empty(self, full_sync_sender):
        """Test getting keys from empty cache."""
        sender, _, _, _ = full_sync_sender

        keys = sender._get_all_hot_cache_keys()

        assert keys == []

    def test_get_all_hot_cache_keys(self):
        """Test getting keys from populated cache."""
        config = create_test_config()
        worker = MockWorker()
        engine = MockLMCacheEngine()

        # Create keys
        test_keys = [create_test_key(i) for i in range(10)]
        backend = MockLocalCPUBackend(keys=test_keys)

        sender = FullSyncSender(
            config=config,
            worker=worker,
            lmcache_engine=engine,
            local_cpu_backend=backend,
        )

        keys = sender._get_all_hot_cache_keys()

        assert len(keys) == 10
        assert all(isinstance(k, int) for k in keys)


class TestFullSyncSenderSendBatch:
    """Test cases for sending sync batch messages."""

    def test_send_sync_batch(self, full_sync_sender):
        """Test sending a batch message."""
        sender, worker, _, _ = full_sync_sender

        sender._send_sync_batch("sync_123", 0, [1, 2, 3, 4, 5])

        assert len(worker.messages) == 1
        msg = worker.messages[0]
        assert isinstance(msg, FullSyncBatchMsg)
        assert msg.sync_id == "sync_123"
        assert msg.batch_id == 0
        assert msg.keys == [1, 2, 3, 4, 5]

    def test_send_sync_batch_multiple(self, full_sync_sender):
        """Test sending multiple batch messages."""
        sender, worker, _, _ = full_sync_sender

        sender._send_sync_batch("sync_123", 0, [1, 2, 3])
        sender._send_sync_batch("sync_123", 1, [4, 5, 6])
        sender._send_sync_batch("sync_123", 2, [7, 8, 9])

        assert len(worker.messages) == 3
        assert worker.messages[0].batch_id == 0
        assert worker.messages[1].batch_id == 1
        assert worker.messages[2].batch_id == 2


class TestFullSyncSenderSendEnd:
    """Test cases for sending sync end message."""

    def test_send_sync_end(self, full_sync_sender):
        """Test sending end message."""
        sender, worker, _, _ = full_sync_sender

        sender._send_sync_end("sync_123", 1000)

        assert len(worker.messages) == 1
        msg = worker.messages[0]
        assert isinstance(msg, FullSyncEndMsg)
        assert msg.sync_id == "sync_123"
        assert msg.actual_total_keys == 1000


class TestFullSyncSenderStartSync:
    """Test cases for the full sync process."""

    @pytest.mark.asyncio
    async def test_start_full_sync_empty_cache(self, full_sync_sender):
        """Test full sync with empty cache."""
        sender, worker, engine, _ = full_sync_sender

        # Set response for status query to allow immediate exit
        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="test", accepted=True),
                FullSyncStatusRetMsg(
                    sync_id="test",
                    is_complete=True,
                    global_progress=1.0,
                    can_exit_freeze=True,
                ),
            ]
        )

        success = await sender.start_full_sync("test_reason")

        assert success is True
        assert sender.is_syncing is False
        # Check freeze mode was entered and exited
        assert engine.freeze_mode is False  # Should be exited

    @pytest.mark.asyncio
    async def test_start_full_sync_with_keys(self):
        """Test full sync with keys in cache."""
        config = create_test_config(batch_size=5)
        worker = MockWorker()
        engine = MockLMCacheEngine()

        # Create 12 keys (will need 3 batches)
        test_keys = [create_test_key(i) for i in range(12)]
        backend = MockLocalCPUBackend(keys=test_keys)

        sender = FullSyncSender(
            config=config,
            worker=worker,
            lmcache_engine=engine,
            local_cpu_backend=backend,
        )

        # Set responses
        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="test", accepted=True),
                FullSyncStatusRetMsg(
                    sync_id="test",
                    is_complete=True,
                    global_progress=1.0,
                    can_exit_freeze=True,
                ),
            ]
        )

        success = await sender.start_full_sync("test_reason")

        assert success is True

        # Check messages: 3 batch messages + 1 end message
        batch_msgs = [m for m in worker.messages if isinstance(m, FullSyncBatchMsg)]
        end_msgs = [m for m in worker.messages if isinstance(m, FullSyncEndMsg)]

        assert len(batch_msgs) == 3
        assert len(end_msgs) == 1

        # Verify batch contents
        all_keys = []
        for msg in batch_msgs:
            all_keys.extend(msg.keys)
        assert len(all_keys) == 12

    @pytest.mark.asyncio
    async def test_start_full_sync_already_syncing(self, full_sync_sender):
        """Test that concurrent sync is prevented."""
        sender, worker, engine, _ = full_sync_sender

        # Manually set syncing flag
        sender._is_syncing = True

        success = await sender.start_full_sync("test_reason")

        assert success is False

    @pytest.mark.asyncio
    async def test_start_full_sync_start_rejected(self, full_sync_sender):
        """Test handling of rejected sync start."""
        sender, worker, engine, _ = full_sync_sender

        # Set rejection response for all retries
        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="test", accepted=False, error_msg="Busy"),
                FullSyncStartRetMsg(sync_id="test", accepted=False, error_msg="Busy"),
                FullSyncStartRetMsg(sync_id="test", accepted=False, error_msg="Busy"),
            ]
        )

        success = await sender.start_full_sync("test_reason")

        assert success is False
        # Freeze mode should be exited on failure
        assert engine.freeze_mode is False

    @pytest.mark.asyncio
    async def test_start_full_sync_retry_success(self, full_sync_sender):
        """Test successful sync after retry."""
        sender, worker, engine, _ = full_sync_sender

        # First attempt fails, second succeeds
        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="test", accepted=False, error_msg="Busy"),
                FullSyncStartRetMsg(sync_id="test", accepted=True),
                FullSyncStatusRetMsg(
                    sync_id="test",
                    is_complete=True,
                    global_progress=1.0,
                    can_exit_freeze=True,
                ),
            ]
        )

        success = await sender.start_full_sync("test_reason")

        assert success is True

    @pytest.mark.asyncio
    async def test_start_full_sync_freeze_mode(self, full_sync_sender):
        """Test that freeze mode is properly managed."""
        sender, worker, engine, _ = full_sync_sender

        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="test", accepted=True),
                FullSyncStatusRetMsg(
                    sync_id="test",
                    is_complete=True,
                    global_progress=1.0,
                    can_exit_freeze=True,
                ),
            ]
        )

        # Track freeze mode changes
        freeze_changes = []

        def track_freeze(mode):
            freeze_changes.append(mode)
            engine.freeze_mode = mode

        engine.set_hot_cache_freeze_mode = track_freeze

        await sender.start_full_sync("test_reason")

        # Should have entered and exited freeze mode
        assert True in freeze_changes
        assert False in freeze_changes
        # Final state should be not frozen
        assert freeze_changes[-1] is False


class TestFullSyncSenderSendSyncStart:
    """Test cases for _send_sync_start method."""

    @pytest.mark.asyncio
    async def test_send_sync_start_success(self, full_sync_sender):
        """Test successful sync start."""
        sender, worker, _, _ = full_sync_sender

        worker.set_responses(
            [
                FullSyncStartRetMsg(sync_id="sync_123", accepted=True),
            ]
        )

        ret = await sender._send_sync_start("sync_123", 100, 5)

        assert ret is not None
        assert ret.accepted is True

    @pytest.mark.asyncio
    async def test_send_sync_start_rejected(self, full_sync_sender):
        """Test rejected sync start."""
        sender, worker, _, _ = full_sync_sender

        worker.set_responses(
            [
                FullSyncStartRetMsg(
                    sync_id="sync_123",
                    accepted=False,
                    error_msg="Already syncing",
                ),
            ]
        )

        ret = await sender._send_sync_start("sync_123", 100, 5)

        assert ret is not None
        assert ret.accepted is False
        assert ret.error_msg == "Already syncing"


class TestFullSyncSenderQueryStatus:
    """Test cases for _query_sync_status method."""

    @pytest.mark.asyncio
    async def test_query_sync_status_complete(self, full_sync_sender):
        """Test query status for complete sync."""
        sender, worker, _, _ = full_sync_sender

        worker.set_responses(
            [
                FullSyncStatusRetMsg(
                    sync_id="sync_123",
                    is_complete=True,
                    global_progress=0.85,
                    can_exit_freeze=True,
                ),
            ]
        )

        ret = await sender._query_sync_status("sync_123")

        assert ret is not None
        assert ret.is_complete is True
        assert ret.global_progress == 0.85
        assert ret.can_exit_freeze is True

    @pytest.mark.asyncio
    async def test_query_sync_status_incomplete(self, full_sync_sender):
        """Test query status for incomplete sync."""
        sender, worker, _, _ = full_sync_sender

        worker.set_responses(
            [
                FullSyncStatusRetMsg(
                    sync_id="sync_123",
                    is_complete=False,
                    global_progress=0.3,
                    can_exit_freeze=False,
                ),
            ]
        )

        ret = await sender._query_sync_status("sync_123")

        assert ret is not None
        assert ret.is_complete is False
        assert ret.global_progress == 0.3
        assert ret.can_exit_freeze is False
