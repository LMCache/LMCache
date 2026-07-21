# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for DramL2Adapter.

Tests the in-DRAM L2 adapter independently (no serde, no QAT).
Validates store/load/lookup/delete/eviction/capacity semantics.
"""

# Standard
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.dram_l2_adapter import (
    DramL2Adapter,
    DramL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


# =============================================================================
# Helpers
# =============================================================================


class _RecordingListener(L2AdapterListener):
    """Listener that records all events for inspection in tests."""

    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self.deleted.append(list(keys))


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: int = 0x42) -> TensorMemoryObj:
    """Create a TensorMemoryObj with known byte content.

    Uses uint8 so byte_array content is predictable.
    """
    raw_data = torch.full((size,), fill_value, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=0,
        phy_size=size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 2.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def adapter():
    """DramL2Adapter with ~1KB capacity."""
    config = DramL2AdapterConfig(max_size_gb=1e-6)  # ~1KB
    a = DramL2Adapter(config)
    yield a
    a.close()


@pytest.fixture
def large_adapter():
    """DramL2Adapter with 1MB capacity."""
    config = DramL2AdapterConfig(max_size_gb=0.001)  # ~1MB
    a = DramL2Adapter(config)
    yield a
    a.close()


# =============================================================================
# Store Tests
# =============================================================================


class TestStore:
    def test_store_signals_event_fd(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        fd = large_adapter.get_store_event_fd()

        large_adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(fd)

    def test_store_pop_completed(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj(size=100)
        fd = large_adapter.get_store_event_fd()

        task_id = large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(fd)

        results = large_adapter.pop_completed_store_tasks()
        assert task_id in results
        assert results[task_id].is_successful()
        assert results[task_id].bytes_transferred() == 100

    def test_store_duplicate_key_skipped(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj(size=100)
        fd = large_adapter.get_store_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(fd)
        large_adapter.pop_completed_store_tasks()

        # Store same key again
        task_id = large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(fd)
        results = large_adapter.pop_completed_store_tasks()
        # Duplicate → 0 bytes stored
        assert results[task_id].bytes_transferred() == 0

    def test_store_capacity_exceeded(self, adapter):
        """When capacity is ~1KB, a 2KB object should be skipped."""
        key = create_object_key(1)
        obj = create_memory_obj(size=2048)  # 2KB > ~1KB capacity
        fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task([key], [obj])
        wait_for_event_fd(fd)
        results = adapter.pop_completed_store_tasks()
        assert results[task_id].bytes_transferred() == 0


# =============================================================================
# Lookup Tests
# =============================================================================


class TestLookup:
    def test_lookup_hit(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = large_adapter.get_store_event_fd()
        lookup_fd = large_adapter.get_lookup_and_lock_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd)

        task_id = large_adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd)

        bitmap = large_adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

    def test_lookup_miss(self, large_adapter):
        key = create_object_key(999)
        lookup_fd = large_adapter.get_lookup_and_lock_event_fd()

        task_id = large_adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd)

        bitmap = large_adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False


# =============================================================================
# Load Tests
# =============================================================================


class TestLoad:
    def test_store_then_load_roundtrip(self, large_adapter):
        """Data stored can be loaded back correctly."""
        key = create_object_key(1)
        src = create_memory_obj(size=128, fill_value=0xAB)
        store_fd = large_adapter.get_store_event_fd()
        load_fd = large_adapter.get_load_event_fd()

        large_adapter.submit_store_task([key], [src])
        wait_for_event_fd(store_fd)

        # Create empty destination buffer
        dst = create_memory_obj(size=128, fill_value=0x00)
        task_id = large_adapter.submit_load_task([key], [dst])
        wait_for_event_fd(load_fd)

        bitmap = large_adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

        # Verify content matches
        assert bytes(dst.byte_array) == bytes(src.byte_array)

    def test_load_missing_key(self, large_adapter):
        key = create_object_key(999)
        dst = create_memory_obj(size=128, fill_value=0x00)
        load_fd = large_adapter.get_load_event_fd()

        task_id = large_adapter.submit_load_task([key], [dst])
        wait_for_event_fd(load_fd)

        bitmap = large_adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False

    def test_load_multiple_keys_partial_hit(self, large_adapter):
        """Load with mixed hit/miss keys."""
        key1 = create_object_key(1)
        key2 = create_object_key(2)
        src = create_memory_obj(size=64, fill_value=0xCC)
        store_fd = large_adapter.get_store_event_fd()
        load_fd = large_adapter.get_load_event_fd()

        # Only store key1
        large_adapter.submit_store_task([key1], [src])
        wait_for_event_fd(store_fd)

        dst1 = create_memory_obj(size=64, fill_value=0x00)
        dst2 = create_memory_obj(size=64, fill_value=0x00)
        task_id = large_adapter.submit_load_task([key1, key2], [dst1, dst2])
        wait_for_event_fd(load_fd)

        bitmap = large_adapter.query_load_result(task_id)
        assert bitmap.test(0) is True  # key1 hit
        assert bitmap.test(1) is False  # key2 miss
        assert bytes(dst1.byte_array) == bytes(src.byte_array)


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    def test_delete_removes_key(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj(size=100)
        store_fd = large_adapter.get_store_event_fd()
        lookup_fd = large_adapter.get_lookup_and_lock_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd)

        large_adapter.delete([key])

        # Lookup should miss now
        task_id = large_adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd)
        bitmap = large_adapter.query_lookup_and_lock_result(task_id)
        assert bitmap.test(0) is False

    def test_delete_nonexistent_key_no_error(self, large_adapter):
        """Deleting a key that doesn't exist should not raise."""
        large_adapter.delete([create_object_key(999)])


# =============================================================================
# Listener Tests
# =============================================================================


class TestListener:
    def test_store_notifies_listener(self, large_adapter):
        listener = _RecordingListener()
        large_adapter.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = large_adapter.get_store_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd)

        assert len(listener.stored) == 1
        assert key in listener.stored[0]

    def test_delete_notifies_listener(self, large_adapter):
        listener = _RecordingListener()
        large_adapter.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = large_adapter.get_store_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd)

        large_adapter.delete([key])
        assert len(listener.deleted) == 1
        assert key in listener.deleted[0]


# =============================================================================
# Report Status Tests
# =============================================================================


class TestReportStatus:
    def test_report_status_empty(self, large_adapter):
        status = large_adapter.report_status()
        assert status["is_healthy"] is True
        assert status["type"] == "DramL2Adapter"
        assert status["stored_object_count"] == 0
        assert status["current_size_bytes"] == 0

    def test_report_status_after_store(self, large_adapter):
        key = create_object_key(1)
        obj = create_memory_obj(size=200)
        store_fd = large_adapter.get_store_event_fd()

        large_adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd)

        status = large_adapter.report_status()
        assert status["stored_object_count"] == 1
        assert status["current_size_bytes"] == 200
