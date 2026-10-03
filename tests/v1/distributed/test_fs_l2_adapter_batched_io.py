# SPDX-License-Identifier: Apache-2.0
"""Tests for batched filesystem L2 store, lookup, and load operations."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
import asyncio
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters import fs_l2_adapter as fs_module
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])
_TIMEOUT = 5.0


class _MemoryObj:
    def __init__(self, size: int) -> None:
        self.byte_array = bytearray(size)


class _AsyncGate:
    """Hold async calls until all expected calls have started."""

    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._count = 0
        self._lock = threading.Lock()
        self.all_started = threading.Event()
        self.release = threading.Event()

    async def wait(self) -> None:
        with self._lock:
            self._count += 1
            if self._count == self._expected:
                self.all_started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.001)


def _key(index: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=index.to_bytes(32, "big"),
        model_name="test-model",
        kv_rank=0,
        object_group_id=0,
    )


def _objects(payloads: list[bytes]) -> list[_MemoryObj]:
    objects = [_MemoryObj(len(payload)) for payload in payloads]
    for obj, payload in zip(objects, payloads, strict=True):
        obj.byte_array[:] = payload
    return objects


def _as_memory_objects(objects: list[_MemoryObj]) -> list[MemoryObj]:
    return cast(list[MemoryObj], objects)


def _wait_store(
    adapter: FSL2Adapter,
    task_id: int,
    timeout: float = _TIMEOUT,
) -> L2StoreResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = adapter.pop_completed_store_tasks().get(task_id)
        if result is not None:
            return result
        time.sleep(0.001)
    raise TimeoutError(f"store task {task_id} did not complete")


def _wait_bitmap(
    query: Callable[[int], Bitmap | None],
    task_id: int,
    timeout: float = _TIMEOUT,
) -> Bitmap:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = query(task_id)
        if result is not None:
            return result
        time.sleep(0.001)
    raise TimeoutError(f"task {task_id} did not complete")


def _store(
    adapter: FSL2Adapter,
    keys: list[ObjectKey],
    payloads: list[bytes],
) -> L2StoreResult:
    objects = _objects(payloads)
    task_id = adapter.submit_store_task(keys, _as_memory_objects(objects))
    return _wait_store(adapter, task_id)


def _lookup(adapter: FSL2Adapter, keys: list[ObjectKey]) -> Bitmap:
    task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    return _wait_bitmap(adapter.query_lookup_and_lock_result, task_id)


def _load(
    adapter: FSL2Adapter,
    keys: list[ObjectKey],
    sizes: list[int],
) -> tuple[list[_MemoryObj], Bitmap]:
    objects = [_MemoryObj(size) for size in sizes]
    task_id = adapter.submit_load_task(keys, _as_memory_objects(objects))
    bitmap = _wait_bitmap(adapter.query_load_result, task_id)
    return objects, bitmap


def _bits(bitmap: Bitmap, size: int) -> list[bool]:
    return [bool(bitmap.test(index)) for index in range(size)]


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    instance = FSL2Adapter(
        FSL2AdapterConfig(
            base_path=str(tmp_path),
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
    )
    try:
        yield instance
    finally:
        instance.close()


class TestBatchedIO:
    def test_single_key_round_trip(self, adapter: FSL2Adapter) -> None:
        key = _key(1)
        payload = b"single-key-payload"

        store_result = _store(adapter, [key], [payload])
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == len(payload)
        assert _bits(_lookup(adapter, [key]), 1) == [True]

        loaded, bitmap = _load(adapter, [key], [len(payload)])
        assert _bits(bitmap, 1) == [True]
        assert bytes(loaded[0].byte_array) == payload

    def test_round_trip_preserves_order(self, adapter: FSL2Adapter) -> None:
        keys = [_key(index) for index in range(4)]
        payloads = [
            b"first",
            b"second-payload",
            bytes(range(32)),
            b"last" * 17,
        ]

        store_result = _store(adapter, keys, payloads)
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == sum(map(len, payloads))
        assert _bits(_lookup(adapter, keys), len(keys)) == [True] * len(keys)

        loaded, bitmap = _load(adapter, keys, [len(payload) for payload in payloads])
        assert _bits(bitmap, len(keys)) == [True] * len(keys)
        assert [bytes(obj.byte_array) for obj in loaded] == payloads

    def test_store_starts_unique_keys_concurrently(self, adapter: FSL2Adapter) -> None:
        keys = [_key(1), _key(2)]
        payloads = [b"a" * 64, b"b" * 64]
        objects = _objects(payloads)
        gate = _AsyncGate(expected=len(keys))
        original_replace = fs_module.aiofiles.os.replace

        async def gated_replace(source: Any, destination: Any) -> None:
            await gate.wait()
            await original_replace(source, destination)

        with patch.object(fs_module.aiofiles.os, "replace", new=gated_replace):
            task_id = adapter.submit_store_task(keys, _as_memory_objects(objects))
            try:
                assert gate.all_started.wait(timeout=_TIMEOUT)
            finally:
                gate.release.set()
            result = _wait_store(adapter, task_id)

        assert result.is_successful()
        assert result.bytes_transferred() == sum(map(len, payloads))

    def test_lookup_starts_keys_concurrently(self, adapter: FSL2Adapter) -> None:
        keys = [_key(1), _key(2)]
        assert _store(adapter, keys, [b"a", b"b"]).is_successful()
        gate = _AsyncGate(expected=len(keys))
        original_exists = fs_module.aiofiles.os.path.exists

        async def gated_exists(path: Any) -> bool:
            await gate.wait()
            return await original_exists(path)

        with patch.object(fs_module.aiofiles.os.path, "exists", new=gated_exists):
            task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
            try:
                assert gate.all_started.wait(timeout=_TIMEOUT)
            finally:
                gate.release.set()
            bitmap = _wait_bitmap(adapter.query_lookup_and_lock_result, task_id)

        assert _bits(bitmap, len(keys)) == [True, True]

    def test_load_starts_unique_keys_concurrently(self, adapter: FSL2Adapter) -> None:
        keys = [_key(1), _key(2)]
        payloads = [b"a" * 64, b"b" * 64]
        assert _store(adapter, keys, payloads).is_successful()
        objects = [_MemoryObj(len(payload)) for payload in payloads]
        gate = _AsyncGate(expected=len(keys))
        original_read = fs_module._async_readinto_full

        async def gated_read(file: Any, buffer: Any) -> int:
            await gate.wait()
            return await original_read(file, buffer)

        with patch.object(fs_module, "_async_readinto_full", new=gated_read):
            task_id = adapter.submit_load_task(keys, _as_memory_objects(objects))
            try:
                assert gate.all_started.wait(timeout=_TIMEOUT)
            finally:
                gate.release.set()
            bitmap = _wait_bitmap(adapter.query_load_result, task_id)

        assert _bits(bitmap, len(keys)) == [True, True]
        assert [bytes(obj.byte_array) for obj in objects] == payloads


class TestDuplicateKeys:
    def test_store_writes_first_payload_once(self, adapter: FSL2Adapter) -> None:
        key = _key(1)
        first = b"first-payload"
        second = b"other-payload"
        replace_calls = 0
        original_replace = fs_module.aiofiles.os.replace

        async def counting_replace(source: Any, destination: Any) -> None:
            nonlocal replace_calls
            replace_calls += 1
            await original_replace(source, destination)

        with patch.object(fs_module.aiofiles.os, "replace", new=counting_replace):
            result = _store(adapter, [key, key], [first, second])

        assert result.is_successful()
        assert result.bytes_transferred() == len(first)
        assert replace_calls == 1
        loaded, bitmap = _load(adapter, [key], [len(first)])
        assert _bits(bitmap, 1) == [True]
        assert bytes(loaded[0].byte_array) == first

    def test_load_reads_once_and_fans_out(self, adapter: FSL2Adapter) -> None:
        key = _key(1)
        payload = bytes(range(64))
        assert _store(adapter, [key], [payload]).is_successful()
        read_calls = 0
        original_read = fs_module._async_readinto_full

        async def counting_read(file: Any, buffer: Any) -> int:
            nonlocal read_calls
            read_calls += 1
            return await original_read(file, buffer)

        with patch.object(fs_module, "_async_readinto_full", new=counting_read):
            loaded, bitmap = _load(adapter, [key, key, key], [len(payload)] * 3)

        assert read_calls == 1
        assert _bits(bitmap, 3) == [True, True, True]
        assert [bytes(obj.byte_array) for obj in loaded] == [payload] * 3


class TestFailureIsolation:
    def test_store_failure_does_not_prevent_other_writes(
        self, adapter: FSL2Adapter
    ) -> None:
        keys = [_key(1), _key(2), _key(3)]
        payloads = [b"first", b"failed", b"third"]
        failed_hash = keys[1].chunk_hash.hex()
        original_replace = fs_module.aiofiles.os.replace

        async def failing_replace(source: Any, destination: Any) -> None:
            if failed_hash in str(destination):
                raise OSError("injected replace failure")
            await original_replace(source, destination)

        with patch.object(fs_module.aiofiles.os, "replace", new=failing_replace):
            result = _store(adapter, keys, payloads)

        assert not result.is_successful()
        assert result.bytes_transferred() == 0
        assert _bits(_lookup(adapter, keys), len(keys)) == [True, False, True]
        loaded, bitmap = _load(
            adapter, [keys[0], keys[2]], [len(payloads[0]), len(payloads[2])]
        )
        assert _bits(bitmap, 2) == [True, True]
        assert [bytes(obj.byte_array) for obj in loaded] == [payloads[0], payloads[2]]

    def test_lookup_failure_is_isolated(self, adapter: FSL2Adapter) -> None:
        present_keys = [_key(1), _key(2), _key(3)]
        assert _store(adapter, present_keys, [b"a", b"b", b"c"]).is_successful()
        keys = [present_keys[0], present_keys[1], _key(99), present_keys[2]]
        failed_hash = present_keys[1].chunk_hash.hex()
        original_exists = fs_module.aiofiles.os.path.exists

        async def failing_exists(path: Any) -> bool:
            if failed_hash in str(path):
                raise OSError("injected lookup failure")
            return await original_exists(path)

        with patch.object(fs_module.aiofiles.os.path, "exists", new=failing_exists):
            bitmap = _lookup(adapter, keys)

        assert _bits(bitmap, len(keys)) == [True, False, False, True]

    def test_load_failure_is_isolated(self, adapter: FSL2Adapter) -> None:
        keys = [_key(1), _key(2), _key(3)]
        payloads = [b"first", b"failed", b"third"]
        assert _store(adapter, keys, payloads).is_successful()
        failed_hash = keys[1].chunk_hash.hex()
        original_open = fs_module.aiofiles.open

        def failing_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if failed_hash in str(file):
                raise OSError("injected open failure")
            return original_open(file, *args, **kwargs)

        with patch.object(fs_module.aiofiles, "open", new=failing_open):
            loaded, bitmap = _load(
                adapter, keys, [len(payload) for payload in payloads]
            )

        assert _bits(bitmap, len(keys)) == [True, False, True]
        assert bytes(loaded[0].byte_array) == payloads[0]
        assert loaded[1].byte_array == bytearray(len(payloads[1]))
        assert bytes(loaded[2].byte_array) == payloads[2]
