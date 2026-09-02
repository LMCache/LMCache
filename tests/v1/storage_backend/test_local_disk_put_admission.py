# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.local_disk_backend import (
    LocalDiskBackend,
    LocalDiskWorker,
)


def _create_key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def _create_worker() -> LocalDiskWorker:
    worker = object.__new__(LocalDiskWorker)
    worker.put_lock = threading.Lock()
    worker.put_tasks = []
    worker.submit_task = MagicMock(  # type: ignore[method-assign]
        return_value=object()
    )
    return worker


def _create_backend(
    *,
    current_cache_size: int,
    max_cache_size: int,
    entries: dict[CacheEngineKey, SimpleNamespace],
) -> LocalDiskBackend:
    backend = object.__new__(LocalDiskBackend)
    backend.disk_worker = _create_worker()
    backend.current_cache_size = current_cache_size
    backend.max_cache_size = max_cache_size
    backend.dict = entries
    backend.disk_lock = threading.Lock()
    backend.cache_policy = MagicMock()
    backend.batched_remove = MagicMock()  # type: ignore[method-assign]
    backend.loop = MagicMock()
    return backend


def _create_memory_obj(size: int) -> MemoryObj:
    memory_obj = MagicMock(spec=MemoryObj)
    memory_obj.tensor = torch.empty(1, dtype=torch.uint8)
    memory_obj.get_physical_size.return_value = size
    return memory_obj


def _metadata(size: int, *, can_evict: bool) -> SimpleNamespace:
    return SimpleNamespace(size=size, can_evict=can_evict)


def test_try_insert_put_task_is_atomic() -> None:
    worker = _create_worker()
    key = _create_key(1)
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    results = [False] * num_threads

    def register(index: int) -> None:
        barrier.wait()
        results[index] = worker.try_insert_put_task(key)

    threads = [
        threading.Thread(target=register, args=(index,)) for index in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert worker.put_tasks == [key]


def test_rejected_put_can_be_retried_after_capacity_is_released() -> None:
    blocker = _create_key(1)
    key = _create_key(2)
    backend = _create_backend(
        current_cache_size=100,
        max_cache_size=100,
        entries={blocker: _metadata(100, can_evict=False)},
    )
    memory_obj = _create_memory_obj(10)

    with patch(
        "lmcache.v1.storage_backend.local_disk_backend.asyncio.run_coroutine_threadsafe"
    ) as schedule:
        backend.submit_put_task(key, memory_obj)

    assert key not in backend.disk_worker.put_tasks
    assert backend.current_cache_size == 100
    cast(MagicMock, backend.cache_policy.get_evict_candidates).assert_not_called()
    cast(MagicMock, backend.batched_remove).assert_not_called()
    cast(MagicMock, memory_obj.ref_count_up).assert_not_called()
    schedule.assert_not_called()

    backend.current_cache_size = 0
    backend.dict.clear()
    with patch(
        "lmcache.v1.storage_backend.local_disk_backend.asyncio.run_coroutine_threadsafe"
    ) as schedule:
        backend.submit_put_task(key, memory_obj)

    assert key in backend.disk_worker.put_tasks
    cast(MagicMock, backend.cache_policy.update_on_put).assert_called_once_with(key)
    cast(MagicMock, memory_obj.ref_count_up).assert_called_once_with()
    schedule.assert_called_once()


def test_insufficient_evictable_capacity_does_not_remove_resident_keys() -> None:
    evictable = _create_key(1)
    pinned = _create_key(2)
    key = _create_key(3)
    entries = {
        evictable: _metadata(20, can_evict=True),
        pinned: _metadata(80, can_evict=False),
    }
    original_entries = entries.copy()
    backend = _create_backend(
        current_cache_size=100,
        max_cache_size=100,
        entries=entries,
    )
    memory_obj = _create_memory_obj(60)

    with patch(
        "lmcache.v1.storage_backend.local_disk_backend.asyncio.run_coroutine_threadsafe"
    ) as schedule:
        backend.submit_put_task(key, memory_obj)

    assert backend.dict == original_entries
    assert backend.current_cache_size == 100
    assert key not in backend.disk_worker.put_tasks
    cast(MagicMock, backend.cache_policy.get_evict_candidates).assert_not_called()
    cast(MagicMock, backend.batched_remove).assert_not_called()
    cast(MagicMock, backend.cache_policy.update_on_put).assert_not_called()
    cast(MagicMock, memory_obj.ref_count_up).assert_not_called()
    schedule.assert_not_called()


def test_oversized_put_fails_before_registering_or_evicting() -> None:
    resident = _create_key(1)
    key = _create_key(2)
    entries = {resident: _metadata(80, can_evict=True)}
    original_entries = entries.copy()
    backend = _create_backend(
        current_cache_size=80,
        max_cache_size=100,
        entries=entries,
    )
    memory_obj = _create_memory_obj(120)

    with patch(
        "lmcache.v1.storage_backend.local_disk_backend.asyncio.run_coroutine_threadsafe"
    ) as schedule:
        backend.submit_put_task(key, memory_obj)

    assert backend.dict == original_entries
    assert backend.current_cache_size == 80
    assert backend.disk_worker.put_tasks == []
    cast(MagicMock, backend.cache_policy.get_evict_candidates).assert_not_called()
    cast(MagicMock, backend.batched_remove).assert_not_called()
    cast(MagicMock, backend.cache_policy.update_on_put).assert_not_called()
    cast(MagicMock, memory_obj.ref_count_up).assert_not_called()
    schedule.assert_not_called()
