# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from types import SimpleNamespace
from typing import cast

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager


class DummyMemoryObj:
    def __init__(self, ref_count: int = 1):
        self.meta = SimpleNamespace(ref_count=ref_count)
        self.down_calls = 0

    def ref_count_down(self):
        self.meta.ref_count -= 1
        self.down_calls += 1


class DummyAllocatorBackend:
    """Allocator stub that acts as both allocator and storage backend."""

    def __init__(self):
        self.calls = 0

    def get_allocator_backend(self):
        return self

    def batched_submit_put_task(self, keys, objs, transfer_spec=None):
        self.calls += 1
        return None


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
    manager.storage_backends = cast(
        OrderedDict[str, StorageBackendInterface], backends
    )
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

    manager.batched_put(keys, mem_objs)

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

    manager.batched_put(keys, mem_objs)
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

    manager.batched_put(keys, mem_objs)

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

    manager.batched_put(keys, mem_objs)

    f1.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 1
        assert mem.down_calls == 0

    f2.set_result(None)
    for mem in mem_objs:
        assert mem.meta.ref_count == 0
        assert mem.down_calls == 1
