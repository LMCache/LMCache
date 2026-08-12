# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for DynamicNixlStoreL2Adapter with POSIX backend.

Tests cover the L2AdapterInterface contract, dynamic file operations,
persist, secondary lookup, and capacity management.
"""

# Standard
import asyncio
import os
import select
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

nixl = pytest.importorskip("nixl")

# First Party
from lmcache import torch_device_type  # noqa: E402
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.internal_api import (  # noqa: E402
    L1MemoryDesc,
    L2AdapterListener,
)
from lmcache.v1.distributed.l2_adapters import (  # noqa: E402
    nixl_store_dynamic_l2_adapter as dynamic_nixl_module,
)
from lmcache.v1.distributed.l2_adapters.config import PersistConfig  # noqa: E402
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

DynamicNixlStoreL2Adapter = dynamic_nixl_module.DynamicNixlStoreL2Adapter
DynamicNixlStoreL2AdapterConfig = dynamic_nixl_module.DynamicNixlStoreL2AdapterConfig
DynamicNixlStorageAgent = dynamic_nixl_module.DynamicNixlStorageAgent
_object_key_to_filename = dynamic_nixl_module._object_key_to_filename

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


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


# =============================================================================
# Constants
# =============================================================================

PAGE_SIZE = 4096  # 4 KB per page
NUM_BUFFER_PAGES = 20  # pages in the registered memory buffer
MAX_CAPACITY_GB = 0.001  # ~1 MB

if torch_device_type == "xpu":
    pytest.skip(
        (
            "Skip on XPU: in vllm/vllm-openai-xpu:v0.26.0, "
            "NIXL dynamic store backends are unavailable at runtime "
            "(including POSIX), adapter init can fail with "
            "NIXL_ERR_NOT_FOUND, so this suite is not runnable "
            "on XPU in the current test environment."
        ),
        allow_module_level=True,
    )

# =============================================================================
# Test Helpers
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(
    buffer: torch.Tensor,
    page_index: int,
    fill_value: float = 1.0,
    num_pages: int = 1,
) -> TensorMemoryObj:
    """Create a TensorMemoryObj that references page(s) in the registered buffer."""
    obj_size = PAGE_SIZE * num_pages
    start = page_index * PAGE_SIZE
    end = start + obj_size
    num_floats = obj_size // 4

    raw_data = buffer[start:end].view(torch.float32)
    raw_data.fill_(fill_value)

    metadata = MemoryObjMetadata(
        shape=torch.Size([num_floats]),
        dtype=torch.float32,
        address=page_index * PAGE_SIZE,
        phy_size=obj_size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Wait for an event fd to be signaled."""
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
# Test Fixtures
# =============================================================================


@pytest.fixture
def adapter():
    """Create a DynamicNixlStoreL2Adapter with POSIX backend.

    Yields (adapter, buffer) so tests can create memory objects that
    reference pages inside the registered buffer.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_test_")

    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")

    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(),
        size=buffer.numel(),
        align_bytes=PAGE_SIZE,
    )

    config = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
        },
    )
    adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

    yield adpt, buffer, tmp_dir

    adpt.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def adapter_with_persist():
    """Create a DynamicNixlStoreL2Adapter with persist enabled.

    Yields (adapter, buffer, tmp_dir, l1_memory, config) and does NOT call
    close() — tests manage the lifecycle themselves.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_persist_test_")

    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")

    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(),
        size=buffer.numel(),
        align_bytes=PAGE_SIZE,
    )

    config = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
        },
    )
    config.persist_config = PersistConfig(persist_enabled=True)
    adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

    yield adpt, buffer, tmp_dir, l1_memory, config

    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    def test_get_store_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_store_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_lookup_and_lock_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_lookup_and_lock_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_load_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_load_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_event_fds_are_different(self, adapter):
        adpt, _, _ = adapter
        fds = {
            adpt.get_store_event_fd(),
            adpt.get_lookup_and_lock_event_fd(),
            adpt.get_load_event_fd(),
        }
        assert len(fds) == 3


# =============================================================================
# Store Interface Tests
# =============================================================================


class TestStoreInterface:
    def test_submit_store_task_returns_task_id(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_store_task([key], [obj])
        assert isinstance(task_id, int)

    def test_submit_store_task_signals_event_fd(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        assert wait_for_event_fd(adpt.get_store_event_fd())

    def test_pop_completed_store_tasks_returns_completed(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())

        completed = adpt.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id].is_successful()

    def test_store_creates_file_on_disk(self, adapter):
        adpt, buf, tmp_dir = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())

        expected_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert os.path.exists(expected_file)

    def test_submit_multiple_store_tasks_unique_ids(self, adapter):
        adpt, buf, _ = adapter
        key1 = create_object_key(1)
        key2 = create_object_key(2)
        obj1 = create_memory_obj(buf, page_index=0)
        obj2 = create_memory_obj(buf, page_index=1)

        task_id1 = adpt.submit_store_task([key1], [obj1])
        task_id2 = adpt.submit_store_task([key2], [obj2])
        assert task_id1 != task_id2


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    def test_lookup_nonexistent_key_returns_zeros(self, adapter):
        adpt, _, _ = adapter
        key = create_object_key(999)

        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

    def test_lookup_existing_key_returns_ones(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        # Store first
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup
        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        # Unlock
        adpt.submit_unlock([key])

    def test_query_lookup_result_clears_result(self, adapter):
        adpt, _, _ = adapter
        key = create_object_key(1)

        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        result1 = adpt.query_lookup_and_lock_result(task_id)
        result2 = adpt.query_lookup_and_lock_result(task_id)
        assert result1 is not None
        assert result2 is None


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    def test_load_existing_key_copies_data(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=42.0)

        # Store
        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Load into a different page
        load_obj = create_memory_obj(buf, page_index=1, fill_value=0.0)
        task_id = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        # Verify data was copied
        loaded_data = buf[PAGE_SIZE : 2 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded_data == 42.0)

        adpt.submit_unlock([key])

    def test_load_multiple_keys_concurrent(self, adapter):
        """A multi-key load task reads every chunk correctly.

        Exercises the concurrent (gather-based) load loop: store several
        keys, then load them all in a single task and verify each landed in
        its own page with the right data.
        """
        adpt, buf, _ = adapter
        keys = [create_object_key(i) for i in range(1, 4)]
        fills = [11.0, 22.0, 33.0]
        store_objs = [
            create_memory_obj(buf, page_index=i, fill_value=fills[i]) for i in range(3)
        ]

        # Store all three
        adpt.submit_store_task(keys, store_objs)
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Load all three into separate pages (3, 4, 5) in ONE task
        load_objs = [
            create_memory_obj(buf, page_index=3 + i, fill_value=0.0) for i in range(3)
        ]
        task_id = adpt.submit_load_task(keys, load_objs)
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i)
            page = 3 + i
            loaded = buf[page * PAGE_SIZE : (page + 1) * PAGE_SIZE].view(torch.float32)
            assert torch.all(loaded == fills[i])

        adpt.submit_unlock(keys)

    def test_load_partial_failure_marks_only_successful(self, adapter):
        """One chunk's failure must not discard the others in the same task.

        With ``asyncio.gather(..., return_exceptions=True)``, a single failed
        load (here: its backing file is removed) is reported as failed for
        that key while the sibling keys still load successfully.
        """
        adpt, buf, tmp_dir = adapter
        keys = [create_object_key(i) for i in range(1, 4)]
        fills = [11.0, 22.0, 33.0]
        store_objs = [
            create_memory_obj(buf, page_index=i, fill_value=fills[i]) for i in range(3)
        ]

        adpt.submit_store_task(keys, store_objs)
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        task_id = adpt.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Sabotage the middle key's backing file so its load raises.
        os.remove(os.path.join(tmp_dir, _object_key_to_filename(keys[1])))

        load_objs = [
            create_memory_obj(buf, page_index=3 + i, fill_value=0.0) for i in range(3)
        ]
        task_id = adpt.submit_load_task(keys, load_objs)
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)  # loaded ok
        assert not bitmap.test(1)  # file removed -> failed, not crashed
        assert bitmap.test(2)  # loaded ok despite sibling failure

        # The two successful loads landed correct data.
        for i in (0, 2):
            page = 3 + i
            loaded = buf[page * PAGE_SIZE : (page + 1) * PAGE_SIZE].view(torch.float32)
            assert torch.all(loaded == fills[i])

        adpt.submit_unlock(keys)

    def test_query_load_result_clears_result(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=1.0)

        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        load_obj = create_memory_obj(buf, page_index=1)
        task_id = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())

        result1 = adpt.query_load_result(task_id)
        result2 = adpt.query_load_result(task_id)
        assert result1 is not None
        assert result2 is None

        adpt.submit_unlock([key])


# =============================================================================
# Store-Lookup-Load End-to-End Test
# =============================================================================


class TestEndToEnd:
    def test_store_lookup_load_workflow(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=99.0)

        # Store
        store_task = adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        completed = adpt.pop_completed_store_tasks()
        assert completed[store_task].is_successful()

        # Lookup
        lookup_task = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(lookup_task)
        assert bitmap is not None
        assert bitmap.test(0)

        # Load into different page
        load_obj = create_memory_obj(buf, page_index=2, fill_value=0.0)
        load_task = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())
        bitmap = adpt.query_load_result(load_task)
        assert bitmap is not None
        assert bitmap.test(0)

        # Verify
        loaded = buf[2 * PAGE_SIZE : 3 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded == 99.0)

        # Unlock
        adpt.submit_unlock([key])


# =============================================================================
# Eviction / Delete Interface Tests
# =============================================================================


class TestEvictionInterface:
    def test_delete_removes_key(self, adapter):
        adpt, buf, tmp_dir = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Delete
        adpt.delete([key])

        # Lookup should miss
        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

        # File should be removed from disk
        expected_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert not os.path.exists(expected_file)

    def test_delete_skips_pinned_key(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lock
        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Delete should skip pinned key
        adpt.delete([key])

        # Should still be found
        task_id = adpt.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        adpt.submit_unlock([key])
        adpt.submit_unlock([key])

    def test_listener_notified_on_store(self, adapter):
        adpt, buf, _ = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        assert len(listener.stored) == 1
        assert key in listener.stored[0]

    def test_listener_notified_on_delete(self, adapter):
        adpt, buf, _ = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        adpt.delete([key])

        assert len(listener.deleted) == 1
        assert key in listener.deleted[0]


# =============================================================================
# Capacity / Usage Tests
# =============================================================================


class TestCapacity:
    def test_get_usage_empty_is_zero(self, adapter):
        adpt, _, _ = adapter
        usage = adpt.get_usage()
        assert usage.usage_fraction == 0.0

    def test_get_usage_increases_after_store(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage = adpt.get_usage()
        assert usage.usage_fraction > 0.0

    def test_get_usage_decreases_after_delete(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage_before = adpt.get_usage().usage_fraction
        adpt.delete([key])
        usage_after = adpt.get_usage().usage_fraction

        assert usage_after < usage_before

    def test_store_rejected_when_capacity_exceeded(self):
        """Store should stop when max capacity is reached."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_cap_test_")
        try:
            buffer = torch.empty(
                PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
            )
            l1_memory = L1MemoryDesc(
                ptr=buffer.data_ptr(),
                size=buffer.numel(),
                align_bytes=PAGE_SIZE,
            )
            # Very small capacity: 1 page worth of data
            tiny_cap_gb = PAGE_SIZE / (1024**3)
            config = DynamicNixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={
                    "file_path": tmp_dir,
                    "use_direct_io": "false",
                    "max_capacity_gb": str(tiny_cap_gb),
                },
            )
            adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

            # Store first object (should succeed)
            key1 = create_object_key(1)
            obj1 = create_memory_obj(buffer, page_index=0)
            adpt.submit_store_task([key1], [obj1])
            wait_for_event_fd(adpt.get_store_event_fd())

            # Store second object (should be rejected due to capacity)
            key2 = create_object_key(2)
            obj2 = create_memory_obj(buffer, page_index=1)
            adpt.submit_store_task([key2], [obj2])
            wait_for_event_fd(adpt.get_store_event_fd())

            # Only first key should be found
            task_id = adpt.submit_lookup_and_lock_task([key1, key2], {0: _EMPTY_LAYOUT})
            wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
            bitmap = adpt.query_lookup_and_lock_result(task_id)
            assert bitmap is not None
            assert bitmap.test(0)  # key1 found
            assert not bitmap.test(1)  # key2 not found

            adpt.submit_unlock([key1])
            adpt.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_batched_store_reports_failure_when_capacity_exceeded(self):
        """A partial batch should preserve completed stores and report failure."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_cap_test_")
        try:
            buffer = torch.empty(
                PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
            )
            l1_memory = L1MemoryDesc(
                ptr=buffer.data_ptr(),
                size=buffer.numel(),
                align_bytes=PAGE_SIZE,
            )
            tiny_cap_gb = PAGE_SIZE / (1024**3)
            config = DynamicNixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={
                    "file_path": tmp_dir,
                    "use_direct_io": "false",
                    "max_capacity_gb": str(tiny_cap_gb),
                },
            )
            adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

            key1 = create_object_key(1)
            key2 = create_object_key(2)
            objects = [
                create_memory_obj(buffer, page_index=0),
                create_memory_obj(buffer, page_index=1),
            ]

            task_id = adpt.submit_store_task([key1, key2], objects)
            wait_for_event_fd(adpt.get_store_event_fd())
            result = adpt.pop_completed_store_tasks()[task_id]

            assert not result.is_successful()
            assert result.bytes_transferred() == 0
            assert adpt.get_usage().total_bytes_used == PAGE_SIZE

            stored_file = os.path.join(tmp_dir, _object_key_to_filename(key1))
            rejected_file = os.path.join(tmp_dir, _object_key_to_filename(key2))
            assert os.path.exists(stored_file)
            assert not os.path.exists(rejected_file)

            lookup_task = adpt.submit_lookup_and_lock_task([key1, key2], _EMPTY_LAYOUT)
            wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
            bitmap = adpt.query_lookup_and_lock_result(lookup_task)
            assert bitmap is not None
            assert bitmap.test(0)
            assert not bitmap.test(1)

            adpt.submit_unlock([key1])
            adpt.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Persist / Secondary Lookup Tests
# =============================================================================


class TestPersistAndSecondaryLookup:
    def test_persist_keeps_files_on_close(self, adapter_with_persist):
        """With persist_enabled=True, data files remain on disk after close."""
        adpt, buf, tmp_dir, _, _ = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert os.path.exists(data_file)

        adpt.close()

        assert os.path.exists(data_file)

    def test_secondary_lookup_finds_key(self, adapter_with_persist):
        """Lookup finds keys whose files exist on disk via secondary lookup."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0, fill_value=77.0)

        # Store and close (files are kept)
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        # New adapter — secondary lookup discovers the persisted file
        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        # Lookup should find the key via secondary lookup
        task_id = adpt2.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        bitmap = adpt2.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_secondary_lookup_and_load_data(self, adapter_with_persist):
        """After secondary lookup, load returns the same data that was stored."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=55.0)

        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        # Lookup (lazy recover) + load
        task_id = adpt2.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        adpt2.query_lookup_and_lock_result(task_id)

        load_obj = create_memory_obj(buf, page_index=2, fill_value=0.0)
        task_id = adpt2.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt2.get_load_event_fd())
        bitmap = adpt2.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        loaded = buf[2 * PAGE_SIZE : 3 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded == 55.0)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_secondary_lookup_misses_when_file_deleted(self, adapter_with_persist):
        """Secondary lookup returns miss for keys whose files are absent on disk."""
        adpt, buf, tmp_dir, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        # Delete the data file manually
        data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        os.unlink(data_file)

        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        task_id = adpt2.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        bitmap = adpt2.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

        adpt2.close()

    def test_secondary_lookup_usage_updates(self, adapter_with_persist):
        """Secondary lookup populates _total_bytes so get_usage reflects disk files."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage_before = adpt.get_usage().usage_fraction
        adpt.close()

        # Right after init, usage is zero (no eager recovery)
        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)
        usage_initial = adpt2.get_usage().usage_fraction
        assert usage_initial == 0.0

        # After a lookup, the key is populated and usage matches
        task_id = adpt2.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        adpt2.query_lookup_and_lock_result(task_id)

        usage_after = adpt2.get_usage().usage_fraction
        assert usage_after == pytest.approx(usage_before, rel=1e-6)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_close_without_persist_deletes_files(self):
        """With persist_enabled=False, close() deletes all data files."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_cleanup_test_")
        try:
            buffer = torch.empty(
                PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
            )
            l1_memory = L1MemoryDesc(
                ptr=buffer.data_ptr(),
                size=buffer.numel(),
                align_bytes=PAGE_SIZE,
            )
            config = DynamicNixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={
                    "file_path": tmp_dir,
                    "use_direct_io": "false",
                    "max_capacity_gb": str(MAX_CAPACITY_GB),
                },
            )
            config.persist_config = PersistConfig(persist_enabled=False)
            adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

            key = create_object_key(1)
            obj = create_memory_obj(buffer, page_index=0)
            adpt.submit_store_task([key], [obj])
            wait_for_event_fd(adpt.get_store_event_fd())
            adpt.pop_completed_store_tasks()

            data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
            assert os.path.exists(data_file)

            adpt.close()

            assert not os.path.exists(data_file)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# OBJ Backend Tests
# =============================================================================


class _FakeNixlApi:
    """Minimal fake for NIXL API calls used by dynamic OBJ registration."""

    def __init__(self) -> None:
        self.register_memory_calls: list[
            tuple[list[tuple[int, int, int, str]], str]
        ] = []
        self.xfer_desc_calls: list[tuple[list[tuple[int, int, int]], str]] = []
        self.prep_xfer_calls: list[tuple[str, list[tuple[int, int, int]], str]] = []
        self.make_xfer_calls: list[tuple[str, list[int], list[int]]] = []
        self.query_memory_calls: list[
            tuple[list[tuple[int, int, int, str]], str, str]
        ] = []
        self.released_handles: list[str] = []
        self.released_dlists: list[str] = []
        self.deregistered: list[str] = []
        self.query_response = [object()]

    def register_memory(
        self, reg_list: list[tuple[int, int, int, str]], mem_type: str
    ) -> str:
        self.register_memory_calls.append((reg_list, mem_type))
        return f"reg-{len(self.register_memory_calls)}"

    def get_xfer_descs(
        self, xfer_desc: list[tuple[int, int, int]], mem_type: str
    ) -> list[tuple[int, int, int]]:
        self.xfer_desc_calls.append((xfer_desc, mem_type))
        return xfer_desc

    def prep_xfer_dlist(
        self, agent_name: str, xfer_descs: list[tuple[int, int, int]], mem_type: str
    ) -> str:
        self.prep_xfer_calls.append((agent_name, xfer_descs, mem_type))
        return f"xfer-{len(self.prep_xfer_calls)}"

    def make_prepped_xfer(
        self,
        operation: str,
        mem_xfer_handler: str,
        mem_indices: list[int],
        xfer_handler: str,
        storage_indices: list[int],
    ) -> str:
        self.make_xfer_calls.append((operation, mem_indices, storage_indices))
        return f"handle-{len(self.make_xfer_calls)}"

    def transfer(self, handle: str) -> str:
        return "DONE"

    def release_xfer_handle(self, handle: str) -> None:
        self.released_handles.append(handle)

    def release_dlist_handle(self, xfer_handler: str) -> None:
        self.released_dlists.append(xfer_handler)

    def deregister_memory(self, reg_descs: str) -> None:
        self.deregistered.append(reg_descs)

    def query_memory(
        self,
        reg_list: list[tuple[int, int, int, str]],
        backend: str,
        mem_type: str,
    ) -> list[object | None]:
        self.query_memory_calls.append((reg_list, backend, mem_type))
        return self.query_response


def _make_obj_storage_agent(
    backend: str = "OBJ",
) -> tuple[DynamicNixlStorageAgent, _FakeNixlApi]:
    """Create a DynamicNixlStorageAgent shell backed by a fake NIXL API."""
    nixl_api = _FakeNixlApi()
    agent = object.__new__(DynamicNixlStorageAgent)
    agent.backend = backend
    agent.mem_type = "OBJ"
    agent.agent_name = "test-agent"
    agent.mem_xfer_handler = "mem-xfer"
    agent.nixl_agent = nixl_api
    agent._device_id_counter = 0
    agent._device_id_lock = threading.Lock()
    return agent, nixl_api


class _FakeObjDynamicStorageAgent:
    """Fake storage agent for OBJ adapter contract tests."""

    instances: list["_FakeObjDynamicStorageAgent"] = []

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        self.l1_align_bytes = l1_memory_desc.align_bytes
        self.backend = backend
        self.backend_params = backend_params
        self.store_calls: list[tuple[list[int], str, int]] = []
        self.load_calls: list[tuple[list[int], str, int]] = []
        self.exists_calls: list[str] = []
        self.existing_objects: set[str] = set()
        self.closed = False
        _FakeObjDynamicStorageAgent.instances.append(self)

    def get_memory_indices(self, raw_addr: int, mem_size: int) -> list[int]:
        return [
            raw_addr // self.l1_align_bytes + i
            for i in range(mem_size // self.l1_align_bytes)
        ]

    def get_object_key_for_key(self, key: ObjectKey) -> str:
        return _object_key_to_filename(key)

    async def dynamic_store_object(
        self, mem_indices: list[int], object_key: str, page_size: int
    ) -> None:
        self.store_calls.append((mem_indices, object_key, page_size))
        self.existing_objects.add(object_key)

    async def dynamic_load_object(
        self, mem_indices: list[int], object_key: str, page_size: int
    ) -> None:
        self.load_calls.append((mem_indices, object_key, page_size))

    def object_exists(self, object_key: str) -> bool:
        self.exists_calls.append(object_key)
        return object_key in self.existing_objects

    def close(self) -> None:
        self.closed = True


class TestObjectBackends:
    @pytest.mark.parametrize(
        "backend,backend_params",
        [
            ("OBJ", {"bucket": "test-bucket"}),
            (
                "AZURE_BLOB",
                {
                    "account_url": "https://example.blob.core.windows.net",
                    "container_name": "test-container",
                },
            ),
        ],
    )
    def test_config_accepts_object_backends_without_file_params(
        self, backend: str, backend_params: dict[str, str]
    ) -> None:
        config = DynamicNixlStoreL2AdapterConfig.from_dict(
            {
                "backend": backend,
                "backend_params": backend_params,
            }
        )

        assert config.backend == backend
        assert config.backend_params == backend_params

    def test_obj_registration_uses_unique_device_ids(self) -> None:
        agent, nixl_api = _make_obj_storage_agent()

        asyncio.run(agent.dynamic_store_object([3, 4], "obj-key-a", PAGE_SIZE))
        asyncio.run(agent.dynamic_load_object([5], "obj-key-b", PAGE_SIZE))

        assert nixl_api.register_memory_calls == [
            ([(0, PAGE_SIZE * 2, 0, "obj-key-a")], "OBJ"),
            ([(0, PAGE_SIZE, 1, "obj-key-b")], "OBJ"),
        ]
        assert nixl_api.xfer_desc_calls == [
            ([(0, PAGE_SIZE, 0), (PAGE_SIZE, PAGE_SIZE, 0)], "OBJ"),
            ([(0, PAGE_SIZE, 1)], "OBJ"),
        ]
        assert nixl_api.make_xfer_calls == [
            ("WRITE", [3, 4], [0, 1]),
            ("READ", [5], [0]),
        ]
        assert nixl_api.released_handles == ["handle-1", "handle-2"]
        assert nixl_api.released_dlists == ["xfer-1", "xfer-2"]
        assert nixl_api.deregistered == ["reg-1", "reg-2"]

    @pytest.mark.parametrize("backend", ["OBJ", "AZURE_BLOB"])
    def test_object_presence_query_uses_nixl_query_memory(self, backend: str) -> None:
        agent, nixl_api = _make_obj_storage_agent(backend)

        assert agent.object_exists("obj-key")
        nixl_api.query_response = [None]
        assert not agent.object_exists("missing-key")

        assert nixl_api.query_memory_calls == [
            ([(0, 0, 0, "obj-key")], backend, "OBJ"),
            ([(0, 0, 0, "missing-key")], backend, "OBJ"),
        ]

    @pytest.mark.parametrize("backend", ["OBJ", "AZURE_BLOB"])
    def test_object_adapter_store_load_and_secondary_lookup(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        _FakeObjDynamicStorageAgent.instances = []
        monkeypatch.setattr(
            dynamic_nixl_module,
            "DynamicNixlStorageAgent",
            _FakeObjDynamicStorageAgent,
        )

        buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8)
        l1_memory = L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=buffer.numel(),
            align_bytes=PAGE_SIZE,
        )
        config = DynamicNixlStoreL2AdapterConfig(
            backend=backend,
            backend_params={"bucket": "test-bucket"},
        )
        adpt = DynamicNixlStoreL2Adapter(config, l1_memory)
        agent = _FakeObjDynamicStorageAgent.instances[0]

        assert not adpt.supports_global_eviction
        assert adpt.get_usage().usage_fraction == -1.0

        key = create_object_key(1)
        store_obj = create_memory_obj(buffer, page_index=0)
        task_id = adpt.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(adpt.get_store_event_fd())
        store_result = adpt.pop_completed_store_tasks()[task_id]

        assert store_result.is_successful()
        assert agent.store_calls == [
            ([0], _object_key_to_filename(key), PAGE_SIZE),
        ]

        load_obj = create_memory_obj(buffer, page_index=1)
        task_id = adpt.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(adpt.get_load_event_fd())
        load_result = adpt.query_load_result(task_id)

        assert load_result is not None
        assert load_result.test(0)
        assert agent.load_calls == [
            ([1], _object_key_to_filename(key), PAGE_SIZE),
        ]

        recovered_key = create_object_key(2)
        agent.existing_objects.add(_object_key_to_filename(recovered_key))
        task_id = adpt.submit_lookup_and_lock_task([recovered_key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        lookup_result = adpt.query_lookup_and_lock_result(task_id)

        assert lookup_result is not None
        assert lookup_result.test(0)
        assert agent.exists_calls == [_object_key_to_filename(recovered_key)]

        adpt.submit_unlock([key, recovered_key])
        adpt.close()
        assert agent.closed
