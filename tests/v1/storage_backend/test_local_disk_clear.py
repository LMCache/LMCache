# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import asyncio
import os
import threading

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.vllm.cache_api import router
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager

pytestmark = pytest.mark.no_shared_allocator
_KV_SHAPE = torch.Size([2, 16, 128])


def _key(index: int) -> CacheEngineKey:
    return CacheEngineKey("clear-test", 1, 0, index, torch.bfloat16)


def _store(
    backend: LocalDiskBackend,
    index: int,
    shape: torch.Size = _KV_SHAPE,
    fmt: MemoryFormat = MemoryFormat.KV_T2D,
) -> tuple[CacheEngineKey, int]:
    obj = backend.local_cpu_backend.allocate(shape, torch.bfloat16, fmt)
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(index)
    size = obj.get_physical_size()
    completed = threading.Event()
    key = _key(index)
    try:
        backend.submit_put_task(
            key, obj, on_complete_callback=lambda _: completed.set()
        )
        assert completed.wait(10), "Disk write did not complete"
    finally:
        obj.ref_count_down()
    assert backend.contains(key)
    return key, size


@pytest.fixture(params=["LRU", "LFU", "FIFO", "MRU"])
def disk_backend(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[LocalDiskBackend]:
    """Use real disk I/O and a small CPU allocator without a GPU service."""
    config = LMCacheEngineConfig.from_defaults(
        local_disk=str(tmp_path),
        max_local_disk_size=1 / 1024,
        cache_policy=request.param,
    )
    cpu = LocalCPUBackend(
        config,
        dst_device="cpu",
        memory_allocator=TensorMemoryAllocator(torch.empty(1024**2, dtype=torch.uint8)),
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    backend = LocalDiskBackend(config, loop, cpu, dst_device="cpu")
    try:
        yield backend
    finally:
        backend.close()
        cpu.close()
        asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result(
            10
        )
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


@pytest.mark.parametrize("locations", [["LocalDiskBackend"], None])
def test_clear_endpoint_removes_disk_files(
    disk_backend: LocalDiskBackend, tmp_path: Path, locations: list[str] | None
) -> None:
    """Both explicit disk clear and all-location clear reach the real backend."""
    keys = [_store(disk_backend, i)[0] for i in range(3)]
    assert len(list(tmp_path.glob("*.pt"))) == 3
    unrelated_file = tmp_path / "unrelated.pt"
    unrelated_file.write_bytes(b"not owned by this backend")
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        LocalCPUBackend=disk_backend.local_cpu_backend, LocalDiskBackend=disk_backend
    )
    engine = SimpleNamespace(
        clear=lambda locations=None, request_configs=None: manager.clear(locations)
    )
    app = FastAPI()
    app.include_router(router)
    app.state.lmcache_adapter = SimpleNamespace(lmcache_engine=engine)
    with TestClient(app) as client:
        params = [("locations", location) for location in locations or []]
        response = client.delete("/cache/clear", params=params)
        assert response.status_code == 200
        assert response.json() == {"status": "success", "num_removed": 3 * 16}
        assert client.delete("/cache/clear", params=params).json()["num_removed"] == 0

    assert not any(disk_backend.contains(key) for key in keys)
    assert list(tmp_path.glob("*.pt")) == [unrelated_file]
    assert disk_backend.usage == 0
    assert disk_backend.current_cache_size == 0


def test_clear_preserves_pinned_entries_and_releases_capacity(
    disk_backend: LocalDiskBackend, tmp_path: Path
) -> None:
    """Pinned KV survives clear; released space can admit subsequent writes."""
    first, size = _store(disk_backend, 1)
    pinned, _ = _store(disk_backend, 2)
    disk_backend.max_cache_size = 2 * size
    assert disk_backend.contains(pinned, pin=True)

    assert disk_backend.clear() == 16
    assert not disk_backend.contains(first)
    assert disk_backend.contains(pinned)
    assert disk_backend.usage == size
    assert disk_backend.current_cache_size == size
    assert len(list(tmp_path.glob("*.pt"))) == 1
    loaded = disk_backend.get_blocking(pinned)
    assert loaded is not None
    try:
        assert loaded.tensor is not None
        assert torch.all(loaded.tensor == 2)
    finally:
        loaded.ref_count_down()

    _store(disk_backend, 3)
    assert disk_backend.unpin(pinned)
    assert disk_backend.clear() == 32
    disk_backend.touch_cache()
    assert disk_backend.current_cache_size == 0
    assert disk_backend.usage == 0
    assert not list(tmp_path.glob("*.pt"))

    # Exceed the budget cumulatively to catch leaked reservations and stale
    # eviction-policy state after repeated clears.
    for index in range(4, 7):
        _store(disk_backend, index)
        assert disk_backend.clear() == 16
    assert disk_backend.current_cache_size == 0
    for index in range(7, 10):
        _store(disk_backend, index)
    assert len(list(tmp_path.glob("*.pt"))) == 2
    assert disk_backend.current_cache_size == 2 * size
    assert disk_backend.clear() == 32
    assert disk_backend.current_cache_size == 0


def test_clear_counts_tokens_using_each_entry_format(
    disk_backend: LocalDiskBackend,
) -> None:
    """Clear returns tokens, matching StorageManager and LocalCPUBackend."""
    _store(disk_backend, 1, torch.Size([2, 8, 128]), MemoryFormat.KV_T2D)
    _store(disk_backend, 2, torch.Size([2, 3, 24, 128]), MemoryFormat.KV_2LTD)
    assert disk_backend.clear() == 32


def test_clear_keeps_inflight_write_reservation(
    disk_backend: LocalDiskBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing residents must not reset space reserved for a pending put."""
    resident, size = _store(disk_backend, 1)
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    write_file = disk_backend.write_file

    def delayed_write(buffer: memoryview, path: str) -> None:
        entered.set()
        assert release.wait(10), "Pending write was not released"
        write_file(buffer, path)

    monkeypatch.setattr(disk_backend, "write_file", delayed_write)
    obj = disk_backend.local_cpu_backend.allocate(
        torch.Size([2, 16, 128]), torch.bfloat16, MemoryFormat.KV_T2D
    )
    assert obj is not None
    pending = _key(2)
    try:
        disk_backend.submit_put_task(
            pending, obj, on_complete_callback=lambda _: completed.set()
        )
        assert entered.wait(10)
        assert disk_backend.clear() == 16
        assert not disk_backend.contains(resident)
        assert disk_backend.current_cache_size == size
    finally:
        release.set()
        obj.ref_count_down()
    assert completed.wait(10)
    assert disk_backend.contains(pending)
    assert disk_backend.current_cache_size == size
    assert disk_backend.clear() == 16
    assert disk_backend.current_cache_size == 0


def test_clear_notifies_evictions(disk_backend: LocalDiskBackend) -> None:
    """Controller listeners receive the same eviction notifications as remove."""
    key, _ = _store(disk_backend, 1)
    sender = MagicMock()
    disk_backend.batched_msg_sender = sender
    assert disk_backend.clear() == 16
    sender.add_kv_op.assert_called_once_with(op_type=OpType.EVICT, key=key.chunk_hash)


def test_clear_unlink_failure_keeps_entry_for_retry(
    disk_backend: LocalDiskBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed unlink preserves the entry, its reservation and policy state."""
    first, size = _store(disk_backend, 1)
    second, _ = _store(disk_backend, 2)
    for key in (first, second):
        assert disk_backend.contains(key, pin=True)
        assert disk_backend.unpin(key)
    remove = os.remove
    calls = 0

    def fail_second_unlink(path: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("injected unlink failure")
        remove(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "remove", fail_second_unlink)
        with pytest.raises(PermissionError, match="injected unlink failure"):
            disk_backend.clear()

    assert not disk_backend.contains(first)
    assert disk_backend.contains(second)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert disk_backend.current_cache_size == size
    assert disk_backend.usage == size
    disk_backend.touch_cache()
    assert disk_backend.clear() == 16
    assert disk_backend.current_cache_size == 0
    assert disk_backend.usage == 0
    assert not list(tmp_path.glob("*.pt"))
