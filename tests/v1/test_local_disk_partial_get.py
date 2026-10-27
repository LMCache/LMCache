# SPDX-License-Identifier: Apache-2.0
"""Local disk GET must release disk_lock when host allocation stops early."""

# Standard
import asyncio
import threading
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


def _key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey("m", 1, 0, chunk_hash, torch.float16)


def _meta(path: str) -> DiskCacheMetadata:
    return DiskCacheMetadata(
        path=path,
        size=1,
        shape=torch.Size([1]),
        dtype=torch.uint8,
        fmt=MemoryFormat.BINARY,
    )


class _Mem:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(cached_positions=None)
        self.byte_array = bytearray(1)
        self.pins = 0

    def pin(self) -> None:
        self.pins += 1


class _Policy:
    def __init__(self) -> None:
        self.hits: list = []

    def update_on_hit(self, key, mapping) -> None:
        self.hits.append(key)


class _CPU:
    def __init__(self, results: list) -> None:
        self._results = list(results)

    def allocate(self, shape, dtype, fmt, busy_loop: bool = False):
        assert busy_loop is False
        return self._results.pop(0)


class _Worker:
    def __init__(self, backend: LocalDiskBackend) -> None:
        self.backend = backend
        self.calls: list[dict] = []

    async def submit_task(self, task_type, task, **kwargs):
        self.calls.append({"task_type": task_type, **kwargs})
        return task(**kwargs)


def _backend(metas: dict, alloc_results: list) -> tuple[LocalDiskBackend, _Worker]:
    backend = object.__new__(LocalDiskBackend)
    backend.disk_lock = threading.Lock()
    backend._disk_prefetch_slots = asyncio.Semaphore(128)
    backend.dict = metas
    backend.cache_policy = _Policy()
    backend.local_cpu_backend = _CPU(alloc_results)
    worker = _Worker(backend)
    backend.disk_worker = worker
    backend.read_file = lambda key, buffer, path: None
    return backend, worker


def _lock_is_free(lock: threading.Lock) -> bool:
    if not lock.acquire(blocking=False):
        return False
    lock.release()
    return True


def test_second_key_alloc_failure_reads_only_the_prefix() -> None:
    first, second = _key(1), _key(2)
    metas = {first: _meta("a"), second: _meta("b")}
    backend, worker = _backend(metas, [_Mem(), None])

    result = asyncio.run(backend.batched_get_non_blocking("lookup", [first, second]))

    assert len(result) == 1
    assert result[0].pins == 1
    assert len(worker.calls) == 1
    call = worker.calls[0]
    assert call["task_type"] == "prefetch"
    assert call["keys"] == [first]
    assert len(call["paths"]) == len(call["keys"]) == len(call["memory_objs"]) == 1
    assert metas[first].pin_count == 0
    assert metas[second].pin_count == 0
    assert _lock_is_free(backend.disk_lock)


def test_first_key_alloc_failure_skips_disk_read() -> None:
    first, second = _key(1), _key(2)
    metas = {first: _meta("a"), second: _meta("b")}
    backend, worker = _backend(metas, [None])

    result = asyncio.run(backend.batched_get_non_blocking("lookup", [first, second]))

    assert result == []
    assert worker.calls == []
    assert metas[first].pin_count == 0
    assert metas[second].pin_count == 0
    assert _lock_is_free(backend.disk_lock)


def test_missing_key_still_raises_and_releases_the_lock() -> None:
    first, second = _key(1), _key(2)
    metas = {first: _meta("a")}
    backend, worker = _backend(metas, [_Mem()])

    with pytest.raises(AssertionError):
        asyncio.run(backend.batched_get_non_blocking("lookup", [first, second]))

    assert worker.calls == []
    assert _lock_is_free(backend.disk_lock)
