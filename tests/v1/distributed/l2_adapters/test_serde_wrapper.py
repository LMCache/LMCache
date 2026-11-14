# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SerdeL2AdapterWrapper failure, shutdown and pass-through paths.

The wrapper runs against a real CPU ``DramL1Manager`` (its temp buffers come from
L1, so leaks are visible through the public usage counters) and two
scriptable fakes: an inner L2 adapter and a ``SerdeProcessor``. Each fake can
make a stage succeed, report failure, raise on submit, or stay pending.

Happy-path round trips through ``StorageManager`` live in
``tests/v1/distributed/serde/test_serde_e2e.py``.
"""

# Standard
from collections.abc import Iterator
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    KeyListPage,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.serde import SerdeProcessor, SerdeTaskId
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import consume_fd, create_event_notifier
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
L1_SLOTS = 8
# Every object in these tests (KV source/destination and serialized temp)
# occupies exactly one 4 KiB slot, so capacity is counted in slots.
KV_LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
SERIALIZED_BYTES = ALIGN
INNER_CAPACITY_BYTES = 1 << 20
INNER_STORED_BYTES = 12345
INNER_LOOKUP_TASK_ID = 77
WAIT_TIMEOUT_S = 5.0


def make_key(index: int) -> ObjectKey:
    """Return a deterministic test key for ``index``."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "serde-wrapper-test", 0)


def commit_objects(
    l1: DramL1Manager, count: int
) -> tuple[list[ObjectKey], list[MemoryObj]]:
    """Write ``count`` resident KV objects into L1 and return them.

    They serve as store sources / load destinations and, in bulk, as filler
    that leaves no room for the wrapper's temp buffers.
    """
    keys = [make_key(i) for i in range(count)]
    reserved = l1.reserve_write(keys, [False] * count, KV_LAYOUT)
    objs: list[MemoryObj] = []
    for k in keys:
        error, obj = reserved[k]
        assert error is L1Error.SUCCESS and obj is not None
        objs.append(obj)
    assert set(l1.finish_write(keys).values()) == {L1Error.SUCCESS}
    return keys, objs


def l1_usage(l1: DramL1Manager) -> tuple[int, int]:
    """Return (used bytes, staging bytes): what a leaked temp would inflate."""
    return l1.get_memory_usage()[0], l1.get_staging_memory_usage()


def wait_for_store_result(
    wrapper: SerdeL2AdapterWrapper, task_id: L2TaskId
) -> L2StoreResult:
    """Block on the wrapper's store eventfd until ``task_id`` completes."""
    fd = wrapper.get_store_event_fd()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while True:
        completed = wrapper.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
        remaining_ms = (deadline - time.monotonic()) * 1000
        if remaining_ms <= 0:
            pytest.fail(f"store task {task_id} never completed ({WAIT_TIMEOUT_S}s)")
        if poller.poll(remaining_ms):
            consume_fd(fd)


def wait_for_load_result(wrapper: SerdeL2AdapterWrapper, task_id: L2TaskId) -> Bitmap:
    """Block on the wrapper's load eventfd until ``task_id`` completes."""
    fd = wrapper.get_load_event_fd()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while True:
        bitmap = wrapper.query_load_result(task_id)
        if bitmap is not None:
            return bitmap
        remaining_ms = (deadline - time.monotonic()) * 1000
        if remaining_ms <= 0:
            pytest.fail(f"load task {task_id} never completed ({WAIT_TIMEOUT_S}s)")
        if poller.poll(remaining_ms):
            consume_fd(fd)


class NullL2Listener(L2AdapterListener):
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        pass

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        pass


class FakeInnerAdapter(L2AdapterInterface):
    """Scriptable inner adapter that records what the wrapper sends it.

    Store and load tasks complete synchronously on submit. Set
    ``store_result`` / ``load_hits`` to ``None`` to leave them pending
    forever, or ``fail_submit_*`` to make the submit call raise.
    """

    def __init__(self) -> None:
        super().__init__(max_capacity_bytes=INNER_CAPACITY_BYTES)
        self.store_result: L2StoreResult | None = L2StoreResult(
            True, INNER_STORED_BYTES
        )
        self.load_hits: list[int] | None = None
        self.fail_submit_store = False
        self.fail_submit_load = False

        self.store_calls: list[list[ObjectKey]] = []
        self.load_calls: list[list[ObjectKey]] = []
        self.store_submitted = threading.Event()
        self.passthrough_calls: list[tuple[str, tuple[object, ...]]] = []
        self.lookup_bitmap = Bitmap(2)

        self._store_notifier = create_event_notifier()
        self._lookup_notifier = create_event_notifier()
        self._load_notifier = create_event_notifier()
        self._task_lock = threading.Lock()
        self._next_id = 0
        self._completed_store: dict[L2TaskId, L2StoreResult] = {}
        self._completed_load: dict[L2TaskId, Bitmap] = {}

    def get_store_event_fd(self) -> int:
        return self._store_notifier.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_notifier.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_notifier.fileno()

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        self.store_calls.append(list(keys))
        if self.fail_submit_store:
            raise RuntimeError("injected inner submit_store_task failure")
        task_id = self._new_task_id()
        if self.store_result is not None:
            with self._task_lock:
                self._completed_store[task_id] = self.store_result
            self._store_notifier.notify()
        self.store_submitted.set()
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._task_lock:
            completed = self._completed_store
            self._completed_store = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        self.passthrough_calls.append(
            ("submit_lookup_and_lock_task", (keys, group_layout_descs))
        )
        return INNER_LOOKUP_TASK_ID

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        self.passthrough_calls.append(("query_lookup_and_lock_result", (task_id,)))
        return self.lookup_bitmap

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        self.passthrough_calls.append(("submit_unlock", (keys,)))

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        self.load_calls.append(list(keys))
        if self.fail_submit_load:
            raise RuntimeError("injected inner submit_load_task failure")
        task_id = self._new_task_id()
        if self.load_hits is not None:
            bitmap = Bitmap(len(keys))
            for index in self.load_hits:
                bitmap.set(index)
            with self._task_lock:
                self._completed_load[task_id] = bitmap
            self._load_notifier.notify()
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._task_lock:
            return self._completed_load.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        self.passthrough_calls.append(("delete", (keys,)))

    def clear(self) -> None:
        self.passthrough_calls.append(("clear", ()))

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        self.passthrough_calls.append(("list_l2_keys", (model_name, page_size, cursor)))
        return KeyListPage(entries=(), next_page_token="next-page")

    def register_listener(self, listener: L2AdapterListener) -> None:
        self.passthrough_calls.append(("register_listener", (listener,)))

    def set_backend_identity(self, name: str, shared: bool = False) -> None:
        self.passthrough_calls.append(("set_backend_identity", (name, shared)))

    def report_status(self) -> dict[str, object]:
        return {"is_healthy": True, "adapter": "fake-inner"}

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)

    def close(self) -> None:
        self._store_notifier.close()
        self._lookup_notifier.close()
        self._load_notifier.close()

    def _new_task_id(self) -> L2TaskId:
        with self._task_lock:
            self._next_id += 1
            return self._next_id


class FakeSerde(SerdeProcessor):
    """Scriptable serde that records the keys of each submitted task.

    ``serialize_result`` / ``deserialize_result`` is what the matching query
    reports (``None`` = pending forever); ``fail_submit_*`` makes the submit
    call raise. Data is not transformed: these tests check task outcomes and
    buffer lifetimes, not bytes.
    """

    def __init__(self) -> None:
        self.serialize_result: bool | None = True
        self.deserialize_result: bool | None = True
        self.fail_submit_serialize = False
        self.fail_submit_deserialize = False

        self.serialize_calls: list[list[ObjectKey]] = []
        self.deserialize_calls: list[list[ObjectKey]] = []
        self.deserialize_submitted = threading.Event()

        self._serialize_notifier = create_event_notifier()
        self._deserialize_notifier = create_event_notifier()
        self._task_lock = threading.Lock()
        self._next_id = 0
        self._serialize_done: dict[SerdeTaskId, bool] = {}
        self._deserialize_done: dict[SerdeTaskId, bool] = {}

    def get_serialize_event_fd(self) -> int:
        return self._serialize_notifier.fileno()

    def get_deserialize_event_fd(self) -> int:
        return self._deserialize_notifier.fileno()

    def submit_serialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> SerdeTaskId:
        self.serialize_calls.append(list(keys))
        if self.fail_submit_serialize:
            raise RuntimeError("injected submit_serialize failure")
        task_id = self._new_task_id()
        if self.serialize_result is not None:
            with self._task_lock:
                self._serialize_done[task_id] = self.serialize_result
            self._serialize_notifier.notify()
        return task_id

    def query_serialize_result(self, task_id: SerdeTaskId) -> bool | None:
        with self._task_lock:
            return self._serialize_done.pop(task_id, None)

    def submit_deserialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> SerdeTaskId:
        self.deserialize_calls.append(list(keys))
        if self.fail_submit_deserialize:
            raise RuntimeError("injected submit_deserialize failure")
        task_id = self._new_task_id()
        if self.deserialize_result is not None:
            with self._task_lock:
                self._deserialize_done[task_id] = self.deserialize_result
            self._deserialize_notifier.notify()
        self.deserialize_submitted.set()
        return task_id

    def query_deserialize_result(self, task_id: SerdeTaskId) -> bool | None:
        with self._task_lock:
            return self._deserialize_done.pop(task_id, None)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return SERIALIZED_BYTES

    def close(self) -> None:
        self._serialize_notifier.close()
        self._deserialize_notifier.close()

    def _new_task_id(self) -> SerdeTaskId:
        with self._task_lock:
            self._next_id += 1
            return self._next_id


@pytest.fixture
def l1(monkeypatch: pytest.MonkeyPatch) -> Iterator[DramL1Manager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    manager = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=L1_SLOTS * ALIGN, use_lazy=False, shm_name=""
            )
        )
    )
    yield manager
    manager.close()


@pytest.fixture
def inner() -> FakeInnerAdapter:
    return FakeInnerAdapter()


@pytest.fixture
def serde() -> FakeSerde:
    return FakeSerde()


@pytest.fixture
def wrapper(
    l1: DramL1Manager, inner: FakeInnerAdapter, serde: FakeSerde
) -> Iterator[SerdeL2AdapterWrapper]:
    adapter = SerdeL2AdapterWrapper(inner, serde, l1)
    yield adapter
    adapter.close()


def test_clear_delegates_to_inner(
    wrapper: SerdeL2AdapterWrapper, inner: FakeInnerAdapter
) -> None:
    wrapper.clear()
    assert inner.passthrough_calls == [("clear", ())]


# =============================================================================
# Store path
# =============================================================================


def test_store_fails_when_l1_has_no_room_for_temp_buffers(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
) -> None:
    filler_keys, filler_objs = commit_objects(l1, L1_SLOTS)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_store_task(filler_keys[:2], filler_objs[:2])

    assert not wait_for_store_result(wrapper, task_id).is_successful()
    assert serde.serialize_calls == []
    assert inner.store_calls == []
    assert l1_usage(l1) == baseline


@pytest.mark.parametrize(
    ("inject", "reaches_inner_store"),
    [
        ("submit_serialize_raises", False),
        ("serialize_reports_failure", False),
        ("inner_submit_store_raises", True),
        ("inner_store_reports_failure", True),
    ],
)
def test_store_failure_at_any_stage_fails_task_and_releases_temps(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
    inject: str,
    reaches_inner_store: bool,
) -> None:
    if inject == "submit_serialize_raises":
        serde.fail_submit_serialize = True
    elif inject == "serialize_reports_failure":
        serde.serialize_result = False
    elif inject == "inner_submit_store_raises":
        inner.fail_submit_store = True
    else:
        inner.store_result = L2StoreResult(False, 0)
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_store_task(keys, objs)

    assert not wait_for_store_result(wrapper, task_id).is_successful()
    assert inner.store_calls == ([keys] if reaches_inner_store else [])
    assert l1_usage(l1) == baseline


def test_store_success_reports_inner_bytes_and_releases_temps(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
) -> None:
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_store_task(keys, objs)

    result = wait_for_store_result(wrapper, task_id)
    assert result.is_successful()
    assert result.bytes_transferred() == INNER_STORED_BYTES
    assert serde.serialize_calls == [keys]
    assert inner.store_calls == [keys]
    assert l1_usage(l1) == baseline


# =============================================================================
# Load path
# =============================================================================


def test_load_fails_when_temp_buffers_cannot_be_allocated(
    l1: DramL1Manager, inner: FakeInnerAdapter, wrapper: SerdeL2AdapterWrapper
) -> None:
    filler_keys, filler_objs = commit_objects(l1, L1_SLOTS)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_load_task(filler_keys[:2], filler_objs[:2])

    assert wait_for_load_result(wrapper, task_id).popcount() == 0
    assert inner.load_calls == []
    assert l1_usage(l1) == baseline


def test_load_deserializes_only_keys_the_inner_adapter_found(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
) -> None:
    inner.load_hits = [0, 2]
    keys, objs = commit_objects(l1, 3)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_load_task(keys, objs)

    assert wait_for_load_result(wrapper, task_id).get_indices_list() == [0, 2]
    assert serde.deserialize_calls == [[keys[0], keys[2]]]
    assert l1_usage(l1) == baseline


def test_load_skips_deserialize_when_inner_adapter_finds_nothing(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
) -> None:
    inner.load_hits = []
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_load_task(keys, objs)

    assert wait_for_load_result(wrapper, task_id).popcount() == 0
    assert serde.deserialize_calls == []
    assert l1_usage(l1) == baseline


@pytest.mark.parametrize(
    "inject",
    [
        "inner_submit_load_raises",
        "submit_deserialize_raises",
        "deserialize_reports_failure",
    ],
)
def test_load_failure_at_any_stage_reports_no_hits_and_releases_temps(
    l1: DramL1Manager,
    inner: FakeInnerAdapter,
    serde: FakeSerde,
    wrapper: SerdeL2AdapterWrapper,
    inject: str,
) -> None:
    inner.load_hits = [0, 1]
    if inject == "inner_submit_load_raises":
        inner.fail_submit_load = True
    elif inject == "submit_deserialize_raises":
        serde.fail_submit_deserialize = True
    else:
        serde.deserialize_result = False
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    task_id = wrapper.submit_load_task(keys, objs)

    # Even though the inner adapter found both keys, a failure after it
    # must not report them as loaded: the destination buffers were never
    # filled.
    assert wait_for_load_result(wrapper, task_id).popcount() == 0
    assert l1_usage(l1) == baseline


# =============================================================================
# Shutdown with work in flight
# =============================================================================


@pytest.mark.parametrize("pending_stage", ["serialize", "inner_store"])
def test_close_releases_temps_of_pending_store(
    l1: DramL1Manager, inner: FakeInnerAdapter, serde: FakeSerde, pending_stage: str
) -> None:
    if pending_stage == "serialize":
        serde.serialize_result = None
    else:
        inner.store_result = None
    wrapper = SerdeL2AdapterWrapper(inner, serde, l1)
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    wrapper.submit_store_task(keys, objs)
    if pending_stage == "inner_store":
        assert inner.store_submitted.wait(WAIT_TIMEOUT_S)
    # The temps are held while the task is in flight, so the final check
    # below proves close() released them rather than never taking them.
    assert l1_usage(l1) != baseline

    wrapper.close()

    assert l1_usage(l1) == baseline


@pytest.mark.parametrize("pending_stage", ["inner_load", "deserialize"])
def test_close_releases_temps_of_pending_load(
    l1: DramL1Manager, inner: FakeInnerAdapter, serde: FakeSerde, pending_stage: str
) -> None:
    if pending_stage == "inner_load":
        inner.load_hits = None
    else:
        inner.load_hits = [0, 1]
        serde.deserialize_result = None
    wrapper = SerdeL2AdapterWrapper(inner, serde, l1)
    keys, objs = commit_objects(l1, 2)
    baseline = l1_usage(l1)

    wrapper.submit_load_task(keys, objs)
    if pending_stage == "deserialize":
        assert serde.deserialize_submitted.wait(WAIT_TIMEOUT_S)
    assert l1_usage(l1) != baseline

    wrapper.close()

    assert l1_usage(l1) == baseline


# =============================================================================
# Pass-through and event fds
# =============================================================================


def test_non_serde_operations_pass_through_to_inner(
    inner: FakeInnerAdapter, wrapper: SerdeL2AdapterWrapper
) -> None:
    keys = [make_key(1), make_key(2)]
    layouts = {0: KV_LAYOUT}
    listener = NullL2Listener()

    assert wrapper.get_lookup_and_lock_event_fd() == (
        inner.get_lookup_and_lock_event_fd()
    )
    assert wrapper.submit_lookup_and_lock_task(keys, layouts) == INNER_LOOKUP_TASK_ID
    assert wrapper.query_lookup_and_lock_result(INNER_LOOKUP_TASK_ID) is (
        inner.lookup_bitmap
    )
    wrapper.submit_unlock(keys)
    wrapper.delete(keys)
    page = wrapper.list_l2_keys(model_name="m", page_size=7, cursor="c")
    assert page.next_page_token == "next-page"
    wrapper.register_listener(listener)
    wrapper.set_backend_identity("fake", shared=True)

    assert inner.passthrough_calls == [
        ("submit_lookup_and_lock_task", (keys, layouts)),
        ("query_lookup_and_lock_result", (INNER_LOOKUP_TASK_ID,)),
        ("submit_unlock", (keys,)),
        ("delete", (keys,)),
        ("list_l2_keys", ("m", 7, "c")),
        ("register_listener", (listener,)),
        ("set_backend_identity", ("fake", True)),
    ]
    # The wrapper declares no capacity of its own, so these values can only
    # come from the inner adapter.
    assert wrapper.supports_global_eviction
    assert wrapper.get_usage().total_capacity_bytes == INNER_CAPACITY_BYTES
    assert wrapper.get_memory_tags() == MemoryTags(
        MemorySegment.STORAGE, MemoryResidency.REMOTE
    )
    assert wrapper.report_status() == {
        "is_healthy": True,
        "adapter": "fake-inner",
        "serde_wrapped": True,
    }


def test_event_fds_are_pairwise_distinct(wrapper: SerdeL2AdapterWrapper) -> None:
    fds = {
        wrapper.get_store_event_fd(),
        wrapper.get_lookup_and_lock_event_fd(),
        wrapper.get_load_event_fd(),
    }
    assert len(fds) == 3
