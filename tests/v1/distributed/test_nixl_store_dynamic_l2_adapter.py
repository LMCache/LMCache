# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for DynamicNixlStoreL2Adapter with file and object backends.

Tests cover the L2AdapterInterface contract, dynamic storage operations,
persist, secondary lookup, capacity management, and OBJ behavior.
"""

# Standard
import asyncio
import inspect
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
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.dynamic_nixl_store_agent import (  # noqa: E402, E501
    DynamicNixlStorageAgent,
    _object_key_to_filename,
    _object_key_to_relpath,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.file_dynamic_nixl_store_agent import (  # noqa: E402, E501
    FileDynamicNixlStorageAgent,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.object_dynamic_nixl_store_agent import (  # noqa: E402, E501
    ObjectDynamicNixlStorageAgent,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (  # noqa: E402, E501
    DynamicNixlStoreL2Adapter,
    DynamicNixlStoreL2AdapterConfig,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


class _RecordingListener(L2AdapterListener):
    """Listener that records all events for inspection in tests."""

    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
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


def test_dynamic_nixl_storage_agent_is_abstract() -> None:
    """The backend-neutral storage agent must only be used via a subclass."""
    assert inspect.isabstract(DynamicNixlStorageAgent)
    assert issubclass(FileDynamicNixlStorageAgent, DynamicNixlStorageAgent)
    assert not inspect.isabstract(FileDynamicNixlStorageAgent)
    assert issubclass(ObjectDynamicNixlStorageAgent, DynamicNixlStorageAgent)
    assert not inspect.isabstract(ObjectDynamicNixlStorageAgent)


def test_file_agent_creates_directory_before_nixl_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory creation failure must not initialize NIXL resources."""
    base_initialized = False

    def fake_base_init(
        self: DynamicNixlStorageAgent,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        del self, device, backend, backend_params, l1_memory_desc
        nonlocal base_initialized
        base_initialized = True

    def raise_permission_error(path: str, exist_ok: bool) -> None:
        del path, exist_ok
        raise PermissionError("cannot create storage directory")

    monkeypatch.setattr(DynamicNixlStorageAgent, "__init__", fake_base_init)
    monkeypatch.setattr(os, "makedirs", raise_permission_error)

    with pytest.raises(PermissionError, match="cannot create storage directory"):
        FileDynamicNixlStorageAgent(
            device="cpu",
            backend="POSIX",
            backend_params={"file_path": "/unwritable", "use_direct_io": "false"},
            l1_memory_desc=L1MemoryDesc(ptr=0, size=0, align_bytes=1),
        )

    assert not base_initialized


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
# Dynamic Storage Agent Tests
# =============================================================================


class TestDynamicStorageAgent:
    """Tests for storage-agent selection and common agent behavior."""

    def test_file_backend_creates_file_storage_agent(self, adapter) -> None:
        """A file backend selects the file-specific storage agent."""
        adpt, _, _ = adapter

        assert isinstance(adpt.nixl_agent, FileDynamicNixlStorageAgent)

    def test_storage_agent_calculates_and_validates_page_indices(self, adapter) -> None:
        """The shared agent validates ranges and returns their page indices."""
        adpt, _, _ = adapter

        assert adpt.nixl_agent.get_memory_indices(PAGE_SIZE, PAGE_SIZE * 2) == [1, 2]

        with pytest.raises(ValueError, match="not aligned"):
            adpt.nixl_agent.get_memory_indices(PAGE_SIZE + 1, PAGE_SIZE)
        with pytest.raises(ValueError, match="not a multiple"):
            adpt.nixl_agent.get_memory_indices(PAGE_SIZE, PAGE_SIZE + 1)


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


# ---- 2-level hex hash-prefix subdir layout (shard_dirs) ----
# base/<hex[:2]>/<hex[2:4]>/filename, used when shard_dirs=true, where hex is
# chunk_hash.hex() (the same value in the filename). create_object_key(n) uses
# IntHash2Bytes(n)=n.to_bytes(4,"big"), so hex is n as 8 zero-padded hex chars.


def test_relpath_is_hex_hash_prefix():
    """Path = <hex[:2]>/<hex[2:4]>/filename, using the chunk-hash hex."""
    key = create_object_key(0x834EBC79)
    h = key.chunk_hash.hex()  # "834ebc79"
    parts = _object_key_to_relpath(key).split(os.sep)
    assert len(parts) == 3
    assert parts[0] == h[:2]  # "83"
    assert parts[1] == h[2:4]  # "4e"
    assert parts[2] == _object_key_to_filename(key)


def test_relpath_subdir_matches_filename_hash_prefix():
    """The two subdir levels are the first four hex chars of the filename hash."""
    for n in (0x834EBC79, 0xABCD1234, 0x00FF10A2):
        key = create_object_key(n)
        h = key.chunk_hash.hex()
        parts = _object_key_to_relpath(key).split(os.sep)
        assert parts[0] == h[:2]
        assert parts[1] == h[2:4]
        # filename embeds the same hex, so the subdir is its leading prefix
        assert parts[2].endswith(f"_{h}.bin")


def test_relpath_spreads_across_all_256_top_dirs():
    """Varying the top hash byte reaches all 256 top-level hex subdirs."""
    tops = {
        _object_key_to_relpath(create_object_key(i << 24)).split(os.sep)[0]
        for i in range(256)
    }
    assert len(tops) == 256  # "00".."ff"


def test_store_uses_hashed_layout_and_loads_back():
    """Integration: a store lands in the 2-level subdir (not flat), and loads back."""
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_hashed_test_")
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")
    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(), size=buffer.numel(), align_bytes=PAGE_SIZE
    )
    config = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
            "shard_dirs": "true",
        },
    )
    adpt = DynamicNixlStoreL2Adapter(config, l1_memory)
    try:
        key = create_object_key(0xABCD1234)
        store_obj = create_memory_obj(buffer, page_index=0, fill_value=42.0)
        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        hashed = os.path.join(tmp_dir, _object_key_to_relpath(key))
        flat = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert os.path.exists(hashed), "store must write to the 2-level subdir"
        assert not os.path.exists(flat), "store must not write to the flat path"

        # Load it back and verify data.
        tid = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(tid)
        load_obj = create_memory_obj(buffer, page_index=1, fill_value=0.0)
        tid = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())
        bitmap = adpt.query_load_result(tid)
        assert bitmap is not None and bitmap.test(0)
        loaded = buffer[PAGE_SIZE : 2 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded == 42.0)
        adpt.submit_unlock([key])
    finally:
        adpt.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Object Backend Tests
# =============================================================================


class _FakeObjectNixlApi:
    """Minimal NIXL fake for object registration and lookup tests."""

    def __init__(self) -> None:
        self.register_memory_calls: list[
            tuple[list[tuple[int, int, int, str]], str]
        ] = []
        self.query_memory_calls: list[
            tuple[list[tuple[int, int, int, str]], str, str]
        ] = []
        self.device_ids: list[int] = []
        self.query_response: list[object | None] = [object()]

    def register_memory(
        self, reg_list: list[tuple[int, int, int, str]], mem_type: str
    ) -> str:
        self.register_memory_calls.append((reg_list, mem_type))
        self.device_ids.append(reg_list[0][2])
        return f"reg-{len(self.register_memory_calls)}"

    def get_xfer_descs(
        self, xfer_desc: list[tuple[int, int, int]], mem_type: str
    ) -> list[tuple[int, int, int]]:
        del mem_type
        return xfer_desc

    def prep_xfer_dlist(
        self, agent_name: str, xfer_descs: list[tuple[int, int, int]], mem_type: str
    ) -> str:
        del agent_name, xfer_descs, mem_type
        return "object-xfer"

    def make_prepped_xfer(
        self,
        direction: str,
        mem_xfer_handler: str,
        mem_indices: list[int],
        storage_xfer_handler: str,
        storage_indices: list[int],
    ) -> str:
        del (
            direction,
            mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )
        return "transfer"

    def transfer(self, handle: str) -> str:
        del handle
        return "DONE"

    def release_xfer_handle(self, handle: str) -> None:
        del handle

    def release_dlist_handle(self, xfer_handler: str) -> None:
        del xfer_handler

    def deregister_memory(self, reg_descs: str) -> None:
        del reg_descs

    def query_memory(
        self,
        reg_list: list[tuple[int, int, int, str]],
        backend: str,
        mem_type: str,
    ) -> list[object | None]:
        self.query_memory_calls.append((reg_list, backend, mem_type))
        return self.query_response


def _make_object_storage_agent() -> tuple[
    ObjectDynamicNixlStorageAgent, _FakeObjectNixlApi
]:
    """Create an object agent shell backed by a fake NIXL API."""
    nixl_api = _FakeObjectNixlApi()
    agent = object.__new__(ObjectDynamicNixlStorageAgent)
    agent.backend = "OBJ"
    agent.agent_name = "test-agent"
    agent.l1_align_bytes = PAGE_SIZE
    agent.mem_xfer_handler = "memory-xfer"
    agent.nixl_agent = nixl_api
    agent._device_id_counter = 0
    agent._device_id_lock = threading.Lock()
    return agent, nixl_api


class _FakeObjectStorageAgent:
    """Object storage agent fake used to verify public adapter behavior."""

    def __init__(self) -> None:
        self.existing_keys: set[ObjectKey] = set()
        self.store_calls: list[tuple[list[int], ObjectKey]] = []
        self.load_calls: list[tuple[list[int], ObjectKey]] = []
        self.delete_calls: list[ObjectKey] = []
        self.cleaned_up = False
        self.closed = False

    def get_memory_indices(self, raw_addr: int, mem_size: int) -> list[int]:
        return list(range(raw_addr // PAGE_SIZE, (raw_addr + mem_size) // PAGE_SIZE))

    async def dynamic_store(self, mem_indices: list[int], key: ObjectKey) -> None:
        self.store_calls.append((mem_indices, key))
        self.existing_keys.add(key)

    async def dynamic_load(self, mem_indices: list[int], key: ObjectKey) -> None:
        self.load_calls.append((mem_indices, key))

    def dynamic_delete(self, key: ObjectKey) -> None:
        self.delete_calls.append(key)

    def get_stored_size(self, key: ObjectKey) -> int | None:
        return 0 if key in self.existing_keys else None

    def cleanup(self) -> None:
        self.cleaned_up = True

    def close(self) -> None:
        self.closed = True


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
def test_object_backend_config_does_not_require_file_parameters(
    backend: str, backend_params: dict[str, str]
) -> None:
    """Object configs accept backend-native parameters without file settings."""
    config = DynamicNixlStoreL2AdapterConfig.from_dict(
        {"backend": backend, "backend_params": backend_params}
    )

    assert config.backend == backend
    assert config.backend_params == backend_params


def test_object_agent_uses_unique_device_ids_and_presence_query() -> None:
    """Object transfers get unique IDs and recover through NIXL presence checks."""
    agent, nixl_api = _make_object_storage_agent()
    first_key = create_object_key(1)
    second_key = create_object_key(2)

    asyncio.run(agent.dynamic_store([3, 4], first_key))
    asyncio.run(agent.dynamic_load([5], second_key))

    assert nixl_api.device_ids == [0, 1]
    assert agent.object_exists("present-key")
    nixl_api.query_response = [None]
    assert not agent.object_exists("missing-key")
    assert nixl_api.query_memory_calls == [
        ([(0, 0, 0, "present-key")], "OBJ", "OBJ"),
        ([(0, 0, 0, "missing-key")], "OBJ", "OBJ"),
    ]


def test_object_adapter_uses_backend_managed_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBJ adapters store, load, and recover without adapter-side deletion."""
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8)
    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(), size=buffer.numel(), align_bytes=PAGE_SIZE
    )
    agent = _FakeObjectStorageAgent()
    monkeypatch.setattr(
        dynamic_nixl_module,
        "_create_dynamic_nixl_storage_agent",
        lambda **kwargs: agent,
    )
    adapter = DynamicNixlStoreL2Adapter(
        DynamicNixlStoreL2AdapterConfig(
            backend="OBJ", backend_params={"bucket": "test-bucket"}
        ),
        l1_memory,
    )
    try:
        assert not adapter.supports_global_eviction
        assert adapter.get_usage().usage_fraction == -1.0

        key = create_object_key(1)
        store_task = adapter.submit_store_task(
            [key], [create_memory_obj(buffer, page_index=0)]
        )
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[store_task].is_successful()

        load_task = adapter.submit_load_task(
            [key], [create_memory_obj(buffer, page_index=1)]
        )
        assert wait_for_event_fd(adapter.get_load_event_fd())
        load_result = adapter.query_load_result(load_task)
        assert load_result is not None and load_result.test(0)

        recovered_key = create_object_key(2)
        agent.existing_keys.add(recovered_key)
        lookup_task = adapter.submit_lookup_and_lock_task(
            [recovered_key], {0: _EMPTY_LAYOUT}
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_result = adapter.query_lookup_and_lock_result(lookup_task)
        assert lookup_result is not None and lookup_result.test(0)

        adapter.delete([key])
        assert agent.delete_calls == []
        adapter.submit_unlock([key, recovered_key])
    finally:
        adapter.close()

    assert agent.closed
    assert not agent.cleaned_up
