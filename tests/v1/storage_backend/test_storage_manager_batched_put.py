# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Callable, List, Optional, cast
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager

OnRelease = Optional[Callable[[], None]]


class DummyMemoryObj:
    def __init__(self, ref_count: int = 1, on_release: OnRelease = None):
        self.meta = SimpleNamespace(ref_count=ref_count)
        self.down_calls = 0
        self._on_release = on_release

    def ref_count_down(self):
        self.meta.ref_count -= 1
        self.down_calls += 1
        if self._on_release is not None:
            self._on_release()


class DummyAllocatorBackend:
    """Allocator stub that acts as both allocator and storage backend."""

    def __init__(self):
        self.calls = 0

    def get_allocator_backend(self):
        return self

    def batched_submit_put_task(self, keys, objs, transfer_spec=None):
        self.calls += 1
        return None


class AltAllocatorBackend(DummyAllocatorBackend):
    """Allocator with different cname to trigger allocate path."""


class DummyBackend:
    """Storage backend stub returning a predetermined value from submit."""

    def __init__(self, allocator, submit_return):
        self.allocator = allocator
        self.submit_return = submit_return
        self.calls = 0

    def get_allocator_backend(self):
        return self.allocator

    def batched_submit_put_task(self, keys, objs, transfer_spec=None):
        self.calls += 1
        return self.submit_return


def _make_keys(count: int):
    return [CacheEngineKey("fmt", "model", 1, 0, chunk_hash=i) for i in range(count)]


def _make_manager(backends: OrderedDict[str, object]) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = cast(OrderedDict[str, StorageBackendInterface], backends)
    # allocator backend keyed by LocalCPUBackend to match StorageManager expectations
    manager.allocator_backend = cast(
        AllocatorBackendInterface, backends["LocalCPUBackend"]
    )
    manager.internal_copy_stream = None
    return manager


def test_batched_put_sync_finalize_once():
    allocator = DummyAllocatorBackend()
    manager = _make_manager(OrderedDict({"LocalCPUBackend": allocator}))

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))

    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1


def test_batched_put_async_single_future_finalizes_after_completion():
    allocator = DummyAllocatorBackend()
    fut = Future()
    async_backend = DummyBackend(allocator, fut)

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "AsyncBackend": async_backend,
            }
        )
    )

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))
    # Still pending because future not completed.
    for mem in mem_objs:
        assert mem.meta.ref_count == 1
        assert mem.down_calls == 0

    fut.set_result(None)

    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1


def test_batched_put_async_list_of_futures_waits_for_all():
    allocator = DummyAllocatorBackend()
    f1, f2 = Future(), Future()
    async_backend = DummyBackend(allocator, [f1, f2])

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "AsyncBackend": async_backend,
            }
        )
    )

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))

    f1.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 1
        assert mem.down_calls == 0

    f2.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1


def test_batched_put_multiple_async_backends_same_cname_waits_for_all():
    allocator = DummyAllocatorBackend()
    f1, f2 = Future(), Future()
    backend_a = DummyBackend(allocator, f1)
    backend_b = DummyBackend(allocator, f2)

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "BackendA": backend_a,
                "BackendB": backend_b,
            }
        )
    )

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))

    f1.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 1
        assert mem.down_calls == 0

    f2.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1


def test_batched_put_async_future_completes_from_thread():
    allocator = DummyAllocatorBackend()
    fut = Future()
    async_backend = DummyBackend(allocator, fut)

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "AsyncBackend": async_backend,
            }
        )
    )

    released = threading.Event()
    mem_obj = DummyMemoryObj(on_release=released.set)
    keys = _make_keys(1)

    manager.batched_put(keys, cast(List[MemoryObj], [mem_obj]))
    assert mem_obj.down_calls == 0

    def _complete():
        time.sleep(0.05)
        fut.set_result(None)

    threading.Thread(target=_complete, daemon=True).start()

    assert released.wait(timeout=1.0), "ref_count_down never triggered"
    assert mem_obj.meta.ref_count == 0
    assert mem_obj.down_calls == 1


def test_batched_put_backend_returns_non_future_treated_as_sync():
    allocator = DummyAllocatorBackend()
    bad_async_return = object()  # lacks add_done_callback -> should be treated sync
    async_backend = DummyBackend(allocator, bad_async_return)

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "AsyncBackend": async_backend,
            }
        )
    )

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))

    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1


def test_batched_put_allocation_failure_still_releases_refs(
    monkeypatch: pytest.MonkeyPatch,
):
    # First Party
    from lmcache.v1.storage_backend import storage_manager as sm_module

    allocator = DummyAllocatorBackend()
    alt_allocator = AltAllocatorBackend()
    failing_backend = DummyBackend(alt_allocator, None)

    allocation_called = False

    def fake_allocate(alloc, keys, objs, stream):
        nonlocal allocation_called
        allocation_called = True
        return [], []

    monkeypatch.setattr(sm_module, "allocate_and_copy_objects", fake_allocate)

    manager = _make_manager(
        OrderedDict(
            {
                "LocalCPUBackend": allocator,
                "FailingBackend": failing_backend,
            }
        )
    )

    mem_objs = [DummyMemoryObj(), DummyMemoryObj()]
    keys = _make_keys(len(mem_objs))

    manager.batched_put(keys, cast(List[MemoryObj], mem_objs))

    assert allocation_called, "allocate_and_copy_objects should be invoked"
    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1
