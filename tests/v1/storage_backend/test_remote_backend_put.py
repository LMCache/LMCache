# SPDX-License-Identifier: Apache-2.0
"""
Tests for RemoteBackend.submit_put_task when a put fails before it is
scheduled on the event loop (serialization error or closed event loop).

A failed submission must release every reference it took, clear the pending
marker for the key so later stores of the same key are attempted, count as a
failed put, and re-raise.
"""

# Standard
from typing import Any, Iterator, cast
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class RefCountedObj:
    """Minimal MemoryObj stand-in that tracks its reference count."""

    def __init__(self) -> None:
        self.ref_count = 1

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1


class PassThroughSerializer:
    """Behaves like the naive serializer: takes a reference, returns the obj."""

    def serialize(self, memory_obj):
        memory_obj.ref_count_up()
        return memory_obj


class FailingSerializer:
    def serialize(self, memory_obj):
        raise ValueError("serialize failed")


class FakeConnection:
    """Remote connector whose put() releases the reference it is handed,
    as RemoteConnector.put() is documented to do."""

    def __init__(self) -> None:
        self.created: list[Any] = []
        self.stored: list[CacheEngineKey] = []

    async def _put(self, key, memory_obj) -> None:
        self.stored.append(key)
        memory_obj.ref_count_down()

    def put(self, key, memory_obj):
        coro = self._put(key, memory_obj)
        self.created.append(coro)
        return coro


@pytest.fixture
def running_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture
def closed_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.close()
    return loop


def make_backend(serializer, loop) -> tuple[RemoteBackend, FakeConnection]:
    conn = FakeConnection()
    backend = object.__new__(RemoteBackend)
    backend.put_tasks = set()
    backend.lock = threading.Lock()
    backend.connection = cast(Any, conn)
    backend.serializer = serializer
    backend.loop = loop
    backend._mla_worker_id_as0_mode = False
    backend._put_failed_count = 0
    return backend, conn


def make_key() -> CacheEngineKey:
    return CacheEngineKey("test_model", 1, 0, 0, torch.bfloat16)


def put_and_wait(backend: RemoteBackend, key, memory_obj) -> None:
    done = threading.Event()
    future = backend.submit_put_task(
        key, memory_obj, on_complete_callback=lambda _: done.set()
    )
    future.result(timeout=5)
    assert done.wait(timeout=5)


def test_put_succeeds_and_releases_refs(running_loop):
    backend, conn = make_backend(PassThroughSerializer(), running_loop)
    key, obj = make_key(), RefCountedObj()

    put_and_wait(backend, key, obj)

    assert conn.stored == [key]
    assert obj.ref_count == 1
    assert not backend.exists_in_put_tasks(key)
    assert backend._put_failed_count == 0


def test_serialize_failure_releases_ref_and_clears_pending(running_loop):
    backend, conn = make_backend(FailingSerializer(), running_loop)
    key, obj = make_key(), RefCountedObj()

    with pytest.raises(ValueError, match="serialize failed"):
        backend.submit_put_task(key, obj)

    assert obj.ref_count == 1
    assert not backend.exists_in_put_tasks(key)
    assert backend._put_failed_count == 1

    # A later store of the same key is attempted normally.
    backend.serializer = PassThroughSerializer()
    put_and_wait(backend, key, obj)
    assert conn.stored == [key]
    assert obj.ref_count == 1


def test_schedule_failure_releases_refs_and_clears_pending(closed_loop, running_loop):
    backend, conn = make_backend(PassThroughSerializer(), closed_loop)
    key, obj = make_key(), RefCountedObj()

    with pytest.raises(RuntimeError):
        backend.submit_put_task(key, obj)

    assert obj.ref_count == 1
    assert not backend.exists_in_put_tasks(key)
    assert backend._put_failed_count == 1
    # The unscheduled put coroutine is closed without ever running.
    assert conn.created[0].cr_frame is None
    assert conn.stored == []

    # A later store of the same key is attempted normally.
    backend.loop = running_loop
    put_and_wait(backend, key, obj)
    assert conn.stored == [key]
    assert obj.ref_count == 1
