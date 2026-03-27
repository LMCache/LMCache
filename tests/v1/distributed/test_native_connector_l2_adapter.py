# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NativeConnectorL2Adapter.

Uses a mock native connector (pure Python) that simulates the pybind-wrapped
C++ IStorageConnector interface, so no Redis or C++ build is needed.
"""

# Standard
import ctypes
import os
import select
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# =============================================================================
# Mock Native Connector (simulates the pybind C++ IStorageConnector interface)
# =============================================================================


class MockNativeConnector:
    """
    Pure-Python mock that implements the same interface as a pybind-wrapped
    C++ IStorageConnector.  Stores data in-memory dicts.

    Methods:
      - event_fd() -> int
      - submit_batch_get(keys, memoryviews) -> int
      - submit_batch_set(keys, memoryviews) -> int
      - submit_batch_exists(keys) -> int
      - drain_completions() -> list[tuple[int, bool, str, list[bool] | None]]
      - close()
    """

    def __init__(self):
        self._efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._store: dict[str, bytes] = {}
        self._next_id = 1
        self._completions: list[tuple[int, bool, str, list[bool] | None]] = []
        self._lock = threading.Lock()
        self._closed = False

    def event_fd(self) -> int:
        return self._efd

    def submit_batch_set(self, keys: list[str], memoryviews: list) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        try:
            for key, mv in zip(keys, memoryviews, strict=False):
                self._store[key] = bytes(mv)
            self._push_completion(fid, True, "", None)
        except Exception as e:
            self._push_completion(fid, False, str(e), None)

        return fid

    def submit_batch_get(self, keys: list[str], memoryviews: list) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        try:
            all_ok = True
            for key, mv in zip(keys, memoryviews, strict=False):
                data = self._store.get(key)
                if data is None:
                    all_ok = False
                    break
                if len(data) != mv.nbytes:
                    all_ok = False
                    break
                # Copy data into the buffer using ctypes (same as C++ void* write)
                dest_ptr = ctypes.c_char_p(
                    ctypes.addressof(ctypes.c_char.from_buffer(mv))
                )
                ctypes.memmove(dest_ptr, data, len(data))
            self._push_completion(fid, all_ok, "", None)
        except Exception as e:
            self._push_completion(fid, False, str(e), None)

        return fid

    def submit_batch_exists(self, keys: list[str]) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        results = [key in self._store for key in keys]
        self._push_completion(fid, True, "", results)

        return fid

    def drain_completions(self) -> list[tuple[int, bool, str, list[bool] | None]]:
        # Drain the eventfd
        try:
            os.eventfd_read(self._efd)
        except BlockingIOError:
            pass

        with self._lock:
            completions = list(self._completions)
            self._completions.clear()
        return completions

    def close(self):
        if not self._closed:
            self._closed = True
            os.close(self._efd)

    def _push_completion(
        self, fid: int, ok: bool, error: str, result_bools: list[bool] | None
    ):
        with self._lock:
            self._completions.append((fid, ok, error, result_bools))
        # Signal the eventfd
        try:
            os.eventfd_write(self._efd, 1)
        except OSError:
            pass


# =============================================================================
# Test Fixtures
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 1024, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def adapter():
    mock_client = MockNativeConnector()
    adp = NativeConnectorL2Adapter(mock_client)
    yield adp
    adp.close()


# =============================================================================
# ObjectKey Serialization Tests
# =============================================================================


class TestObjectKeySerialization:
    def test_serialization_is_deterministic(self):
        key = create_object_key(42, "my_model")
        s1 = _object_key_to_string(key)
        s2 = _object_key_to_string(key)
        assert s1 == s2

    def test_different_keys_produce_different_strings(self):
        k1 = create_object_key(1)
        k2 = create_object_key(2)
        assert _object_key_to_string(k1) != _object_key_to_string(k2)

    def test_serialization_format(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        s = _object_key_to_string(key)
        assert s == "llama@000000ff@00010203"


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    def test_three_distinct_event_fds(self, adapter):
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        assert store_fd >= 0
        assert lookup_fd >= 0
        assert load_fd >= 0
        assert len({store_fd, lookup_fd, load_fd}) == 3


# =============================================================================
# Store Interface Tests
# =============================================================================


class TestStoreInterface:
    def test_submit_store_returns_task_id(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        task_id = adapter.submit_store_task([key], [obj])
        assert isinstance(task_id, int)

    def test_store_signals_event_fd_and_completes(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id] is True

    def test_pop_clears_completed_tasks(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)

        completed1 = adapter.pop_completed_store_tasks()
        assert len(completed1) == 1

        completed2 = adapter.pop_completed_store_tasks()
        assert len(completed2) == 0

    def test_multiple_store_tasks_get_unique_ids(self, adapter):
        store_fd = adapter.get_store_event_fd()
        task_ids = set()
        for i in range(5):
            key = create_object_key(i)
            obj = create_memory_obj(fill_value=float(i))
            task_ids.add(adapter.submit_store_task([key], [obj]))

        assert len(task_ids) == 5

        # Wait for all completions
        completed = {}
        while len(completed) < 5:
            wait_for_event_fd(store_fd, timeout=5.0)
            completed.update(adapter.pop_completed_store_tasks())

        for tid in task_ids:
            assert completed[tid] is True

    def test_batch_store(self, adapter):
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()
        assert completed[task_id] is True


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    def test_lookup_nonexistent_key(self, adapter):
        key = create_object_key(999)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key])
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False

    def test_lookup_existing_key(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store first
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Lookup
        task_id = adapter.submit_lookup_and_lock_task([key])
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

    def test_lookup_mixed_keys(self, adapter):
        existing = create_object_key(1)
        missing = create_object_key(999)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task([existing], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        task_id = adapter.submit_lookup_and_lock_task([existing, missing])
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is False

    def test_query_is_one_shot(self, adapter):
        key = create_object_key(1)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key])
        wait_for_event_fd(lookup_fd, timeout=5.0)

        result1 = adapter.query_lookup_and_lock_result(task_id)
        assert result1 is not None

        result2 = adapter.query_lookup_and_lock_result(task_id)
        assert result2 is None

    def test_query_unknown_task_returns_none(self, adapter):
        assert adapter.query_lookup_and_lock_result(99999) is None


# =============================================================================
# Unlock Interface Tests
# =============================================================================


class TestUnlockInterface:
    def test_unlock_does_not_raise(self, adapter):
        key = create_object_key(1)
        adapter.submit_unlock([key])  # should not raise

    def test_unlock_after_lock(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        task_id = adapter.submit_lookup_and_lock_task([key])
        wait_for_event_fd(lookup_fd, timeout=5.0)
        adapter.query_lookup_and_lock_result(task_id)

        adapter.submit_unlock([key])  # should not raise


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    def test_submit_load_returns_task_id(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        task_id = adapter.submit_load_task([key], [obj])
        assert isinstance(task_id, int)

    def test_load_signals_event_fd(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        adapter.submit_load_task([key], [obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

    def test_load_existing_key_copies_data(self, adapter):
        key = create_object_key(1)
        store_obj = create_memory_obj(size=100, fill_value=42.0)
        load_obj = create_memory_obj(size=100, fill_value=0.0)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store
        adapter.submit_store_task([key], [store_obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Load
        task_id = adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

        # Verify data was copied into the load buffer
        assert torch.all(load_obj.tensor == 42.0)

    def test_load_nonexistent_key_fails(self, adapter):
        key = create_object_key(999)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        # Batch GET failed → no bits set
        assert bitmap.test(0) is False

    def test_query_load_is_one_shot(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        result1 = adapter.query_load_result(task_id)
        assert result1 is not None

        result2 = adapter.query_load_result(task_id)
        assert result2 is None

    def test_query_unknown_task_returns_none(self, adapter):
        assert adapter.query_load_result(99999) is None


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    def test_store_lookup_load_workflow(self, adapter):
        key = create_object_key(1)
        store_obj = create_memory_obj(size=256, fill_value=123.0)
        load_obj = create_memory_obj(size=256, fill_value=0.0)

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store
        store_tid = adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        assert adapter.pop_completed_store_tasks()[store_tid] is True

        # Lookup
        lookup_tid = adapter.submit_lookup_and_lock_task([key])
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True

        # Load
        load_tid = adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)
        bitmap = adapter.query_load_result(load_tid)
        assert bitmap.test(0) is True
        assert torch.all(load_obj.tensor == 123.0)

        # Unlock
        adapter.submit_unlock([key])

    def test_multiple_objects_workflow(self, adapter):
        n = 5
        keys = [create_object_key(i) for i in range(n)]
        store_objs = [
            create_memory_obj(size=64, fill_value=float(i * 10)) for i in range(n)
        ]
        load_objs = [create_memory_obj(size=64, fill_value=0.0) for _ in range(n)]

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store all
        store_tid = adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        assert adapter.pop_completed_store_tasks()[store_tid] is True

        # Lookup all
        lookup_tid = adapter.submit_lookup_and_lock_task(keys)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(n):
            assert bitmap.test(i) is True

        # Load all
        load_tid = adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd, timeout=5.0)
        bitmap = adapter.query_load_result(load_tid)
        for i in range(n):
            assert bitmap.test(i) is True
            assert torch.all(load_objs[i].tensor == float(i * 10))


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    def test_close_does_not_raise(self):
        mock_client = MockNativeConnector()
        adp = NativeConnectorL2Adapter(mock_client)
        adp.close()

    def test_close_after_operations(self):
        mock_client = MockNativeConnector()
        adp = NativeConnectorL2Adapter(mock_client)

        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adp.get_store_event_fd()

        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        adp.close()


# =============================================================================
# Config Tests
# =============================================================================


class TestRESPL2AdapterConfig:
    def test_from_dict_minimal(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            RESPL2AdapterConfig,
        )

        config = RESPL2AdapterConfig.from_dict(
            {
                "type": "resp",
                "host": "localhost",
                "port": 6379,
            }
        )
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.num_workers == 8
        assert config.username == ""
        assert config.password == ""

    def test_from_dict_full(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            RESPL2AdapterConfig,
        )

        config = RESPL2AdapterConfig.from_dict(
            {
                "type": "resp",
                "host": "10.0.0.1",
                "port": 6380,
                "num_workers": 16,
                "username": "user",
                "password": "pass",
            }
        )
        assert config.host == "10.0.0.1"
        assert config.port == 6380
        assert config.num_workers == 16
        assert config.username == "user"
        assert config.password == "pass"

    def test_from_dict_missing_host_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            RESPL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="host"):
            RESPL2AdapterConfig.from_dict({"type": "resp", "port": 6379})

    def test_from_dict_missing_port_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            RESPL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="port"):
            RESPL2AdapterConfig.from_dict({"type": "resp", "host": "localhost"})

    def test_registered_as_resp(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "resp" in get_registered_l2_adapter_types()
