# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration test for runtime Device-DAX L1 reconfiguration
(add/remove lifecycle against live mmap-backed devices). Run with:

    RUN_DEVDAX_L1_INTEGRATION=1 pytest -xvs \
        tests/v1/distributed/test_devdax_l1_reconfigure_integration.py

Uses isolated temporary files by default (same open/fstat/mmap code path);
point at real devices (>=3) with LMCACHE_TEST_DEVDAX_L1_PATHS=/dev/dax0.0,...
Slot size defaults to 2 MiB (DAX mapping granularity); override with
LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
import gc
import os
import tempfile

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxRemoveMode,
)
from lmcache.v1.multiprocess.http_apis.l1_reconfigure_api import router
from lmcache.v1.multiprocess.http_apis.reconfigure_api import router as l2_router

RUN_IT = os.environ.get("RUN_DEVDAX_L1_INTEGRATION") == "1"
REAL_DEVICE_PATHS = [
    path.strip()
    for path in os.environ.get("LMCACHE_TEST_DEVDAX_L1_PATHS", "").split(",")
    if path.strip()
]
USING_REAL_DEVICES = bool(REAL_DEVICE_PATHS)
SLOT_BYTES = int(os.environ.get("LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES", str(2 << 20)))

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not RUN_IT,
        reason="Device-DAX L1 integration test (set RUN_DEVDAX_L1_INTEGRATION=1)",
    ),
]


class _DeviceProvider:
    """Hands out device paths: real ones from LMCACHE_TEST_DEVDAX_L1_PATHS in
    order, else files owned by the fixture's temporary directory."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace
        self._created: list[str] = []
        self._real_index = 0

    def acquire(self, size_in_bytes: int) -> str:
        """Return a supplied device or create a file with the requested capacity."""
        if USING_REAL_DEVICES:
            if self._real_index >= len(REAL_DEVICE_PATHS):
                pytest.skip("not enough real Device-DAX devices provided")
            path = REAL_DEVICE_PATHS[self._real_index]
            self._real_index += 1
            fd = os.open(path, os.O_RDWR)
            try:
                capacity = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if capacity and capacity < size_in_bytes:
                pytest.skip(
                    f"device {path} capacity {capacity} < required {size_in_bytes}"
                )
            return path

        path = os.path.join(self._workspace, f"devdax-arena-{len(self._created)}")
        with open(path, "xb") as handle:
            handle.truncate(size_in_bytes)
        self._created.append(path)
        return path


@pytest.fixture
def devices(tmp_path: Path) -> Iterator[_DeviceProvider]:
    """Yield per-test files, optionally under LMCACHE_TEST_DEVDAX_L1_DIR."""
    with tempfile.TemporaryDirectory(
        dir=os.environ.get("LMCACHE_TEST_DEVDAX_L1_DIR") or tmp_path,
        prefix="lmcache-devdax-",
    ) as workspace:
        yield _DeviceProvider(workspace)


def _layout(num_bytes: int = SLOT_BYTES) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([num_bytes])], dtypes=[torch.uint8])


def _key(seed: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\0" * 28,
        model_name="devdax-l1-reconfig-it",
        kv_rank=0,
    )


def _open_fd_count(path: str) -> int:
    """Return how many of this process's open fds point at ``path``."""
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == path:
                count += 1
        except OSError:
            continue
    return count


def test_runtime_add_and_drain_remove_lifecycle(devices: _DeviceProvider) -> None:
    """Runtime capacity drains without corrupting live entries or leaking fds."""
    primary = devices.acquire(SLOT_BYTES)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=SLOT_BYTES,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=primary,
        )
    )
    try:
        # The pool starts as a single primary arena that is actually mapped.
        statuses = manager.get_arena_statuses()
        assert [status.device_path for status in statuses] == [primary]
        assert statuses[0].is_primary is True
        assert _open_fd_count(primary) > 0

        # Fill the primary arena; the mapping is live shared memory.
        error, primary_objs = manager.allocate(_layout(), count=1)
        assert error == L1Error.SUCCESS
        assert primary_objs[0].raw_tensor is not None
        primary_objs[0].raw_tensor.fill_(0xAB)
        assert torch.all(primary_objs[0].raw_tensor == 0xAB)

        # Primary is full, so allocation fails until we add capacity.
        error, empty = manager.allocate(_layout(), count=1)
        assert error == L1Error.OUT_OF_MEMORY
        assert empty == []

        # Add a device at runtime and confirm it is mapped and serves overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = manager.add_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        assert added.is_primary is False
        assert _open_fd_count(overflow) > 0

        error, overflow_objs = manager.allocate(_layout(), count=2)
        assert error == L1Error.SUCCESS
        assert len(overflow_objs) == 2
        assert overflow_objs[0].raw_tensor is not None
        overflow_objs[0].raw_tensor.fill_(0xCD)
        assert overflow_objs[1].raw_tensor is not None
        overflow_objs[1].raw_tensor.fill_(0xEF)

        used, total = manager.get_memory_usage()
        assert total == 3 * SLOT_BYTES
        assert used == 3 * SLOT_BYTES

        # Drain-remove the overflow arena while its allocations are still live.
        removing = manager.remove_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2

        # A draining arena is excluded from new allocations.
        error, blocked = manager.allocate(_layout(), count=1)
        assert error == L1Error.OUT_OF_MEMORY
        assert torch.all(overflow_objs[0].raw_tensor == 0xCD)
        assert torch.all(overflow_objs[1].raw_tensor == 0xEF)

        # Freeing the arena's last allocation unmaps it automatically.
        manager.free(overflow_objs)
        del overflow_objs
        assert [status.device_path for status in manager.get_arena_statuses()] == [
            primary
        ]
        assert _open_fd_count(overflow) == 0

        # An empty added arena is unmapped immediately on removal.
        third = devices.acquire(SLOT_BYTES)
        manager.add_device(third, SLOT_BYTES)
        removed = manager.remove_device(third, DevDaxRemoveMode.DRAIN)
        assert removed.state == DevDaxArenaState.REMOVED
        assert _open_fd_count(third) == 0

        # The primary arena backs get_l1_memory_desc and cannot be removed.
        with pytest.raises(L1ReconfigureError, match="primary"):
            manager.remove_device(primary)

        manager.free(primary_objs)
        del primary_objs
    finally:
        manager.close()

    # Every device is unmapped once the manager is closed.
    assert _open_fd_count(primary) == 0

    # tmpfs devices are real files, so MAP_SHARED write-through is observable on
    # media; real Device-DAX char devices do not support read()/write() syscalls.
    if not USING_REAL_DEVICES:
        with open(primary, "rb") as handle:
            assert handle.read(SLOT_BYTES) == bytes([0xAB]) * SLOT_BYTES


def test_http_reconfigure_lifecycle(devices: _DeviceProvider) -> None:
    """Exercise HTTP and production StorageManager delegates on live mappings."""
    primary = devices.acquire(SLOT_BYTES)
    storage_manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=SLOT_BYTES,
                    use_lazy=False,
                    shm_name="",
                    align_bytes=4096,
                    devdax_path=primary,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
    )
    try:
        app = FastAPI()
        app.include_router(router)
        app.include_router(l2_router)
        app.state.engine = SimpleNamespace(storage_manager=storage_manager)
        with TestClient(app) as client:
            response = client.get("/reconfigure/dax/l2/status")
            assert response.status_code == 200
            assert response.json() == {
                "enabled": False,
                "backend": "dax",
                "num_adapters": 0,
                "adapters": [],
            }
            response = client.get("/reconfigure/dax/l1/status")
            assert response.status_code == 200
            (primary_status,) = response.json()["arenas"]
            assert primary_status["device_path"] == primary
            assert primary_status["is_primary"] is True

            extra = devices.acquire(SLOT_BYTES)
            response = client.post(
                "/reconfigure/dax/l1/add",
                json={"device_path": extra, "size": SLOT_BYTES},
            )
            assert response.status_code == 200
            assert response.json()["added"]["device_path"] == extra
            assert _open_fd_count(extra) > 0

            response = client.post(
                "/reconfigure/dax/l1/remove",
                json={"device_path": extra},
            )
            assert response.status_code == 200
            (removed,) = response.json()["removed"]["arenas"]
            assert removed["state"] == "removed"
            assert _open_fd_count(extra) == 0

            response = client.get("/reconfigure/dax/l1/status")
            assert response.status_code == 200
            assert [arena["device_path"] for arena in response.json()["arenas"]] == [
                primary
            ]
    finally:
        storage_manager.close()

    assert _open_fd_count(primary) == 0


def test_kv_cache_drain_gates_device_removal(devices: _DeviceProvider) -> None:
    """Removal sentinel via the KV-cache path: a device requested for removal
    stays DRAINING (and readable) while KV entries live on it, and unmaps only
    after the last one is deleted."""
    primary = devices.acquire(SLOT_BYTES)
    l1 = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=SLOT_BYTES,
                use_lazy=False,
                shm_name="",
                align_bytes=4096,
                devdax_path=primary,
            )
        )
    )
    try:
        # KV entry A fills the primary device.
        key_a, key_b, key_c = _key(1), _key(2), _key(3)
        write = l1.reserve_write([key_a], [False], _layout())
        assert write[key_a][0] == L1Error.SUCCESS
        write[key_a][1].tensor.fill_(0xA1)
        assert l1.finish_write([key_a])[key_a] == L1Error.SUCCESS
        del write

        # Add a device at runtime; KV entries B and C land on it as overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = l1.add_devdax_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        for key, fill in ((key_b, 0xB2), (key_c, 0xC3)):
            write = l1.reserve_write([key], [False], _layout())
            assert write[key][0] == L1Error.SUCCESS
            write[key][1].tensor.fill_(fill)
            assert l1.finish_write([key])[key] == L1Error.SUCCESS
            del write
        gc.collect()

        # Per-device usage: the primary holds A, the added device holds B and C.
        statuses = {s.device_path: s for s in l1.get_devdax_arena_statuses()}
        assert statuses[primary].used_bytes == SLOT_BYTES
        assert statuses[primary].active_allocations == 1
        assert statuses[overflow].used_bytes == 2 * SLOT_BYTES
        assert statuses[overflow].active_allocations == 2
        assert statuses[overflow].free_bytes == 0

        # Request removal while B and C are still cached: the device drains.
        removing = l1.remove_devdax_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2
        assert _open_fd_count(overflow) > 0

        # KV cached on a draining device stays readable.
        for key, fill in ((key_b, 0xB2), (key_c, 0xC3)):
            read = l1.reserve_read([key])
            assert read[key][0] == L1Error.SUCCESS
            assert torch.all(read[key][1].tensor == fill)
            assert read[key][1].get_shapes() == _layout().shapes
            assert read[key][1].get_dtypes() == _layout().dtypes
            assert l1.finish_read([key])[key] == L1Error.SUCCESS
            del read
        gc.collect()

        # Deleting one of the two entries keeps the device mapped and draining.
        assert l1.delete([key_b])[key_b] == L1Error.SUCCESS
        gc.collect()
        statuses = {s.device_path: s for s in l1.get_devdax_arena_statuses()}
        assert statuses[overflow].state == DevDaxArenaState.DRAINING
        assert statuses[overflow].active_allocations == 1
        assert _open_fd_count(overflow) > 0

        # Deleting the last entry on the device unmaps it automatically.
        assert l1.delete([key_c])[key_c] == L1Error.SUCCESS
        gc.collect()
        assert [s.device_path for s in l1.get_devdax_arena_statuses()] == [primary]
        assert _open_fd_count(overflow) == 0

        # KV on the remaining device is untouched by the removal.
        read = l1.reserve_read([key_a])
        assert read[key_a][0] == L1Error.SUCCESS
        assert torch.all(read[key_a][1].tensor == 0xA1)
        assert l1.finish_read([key_a])[key_a] == L1Error.SUCCESS
        del read
        assert l1.delete([key_a])[key_a] == L1Error.SUCCESS
        gc.collect()
    finally:
        l1.close()

    assert _open_fd_count(primary) == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_capacity_reuse_and_batch_rollback(
    devices: _DeviceProvider, tmp_path: Path, hybrid: bool
) -> None:
    """OOM rolls back partial batches; freed capacity preserves surviving payloads."""
    slot = SLOT_BYTES
    path = devices.acquire(2 * slot)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=slot if hybrid else 2 * slot,
            devdax_size_in_bytes=2 * slot if hybrid else 0,
            devdax_path=path,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
        )
    )
    try:
        capacity = 3 if hybrid else 2
        error, objects = manager.allocate(_layout(slot), count=capacity)
        assert error == L1Error.SUCCESS
        for index, obj in enumerate(objects):
            assert obj.raw_tensor is not None
            obj.raw_tensor.copy_(torch.arange(slot, dtype=torch.uint8) + index)
            expected = (
                L1BackendType.DRAM if hybrid and index == 0 else L1BackendType.DEVDAX
            )
            assert manager.get_backend_type(obj) == expected
        assert manager.allocate(_layout(slot), count=1) == (L1Error.OUT_OF_MEMORY, [])
        manager.free(objects[-1:])
        objects.pop()
        del obj
        before = manager.get_memory_usage()
        assert manager.allocate(_layout(slot), count=2) == (L1Error.OUT_OF_MEMORY, [])
        assert manager.get_memory_usage() == before
        error, reused = manager.allocate(_layout(slot), count=1)
        assert error == L1Error.SUCCESS
        assert reused[0].raw_tensor is not None
        reused[0].raw_tensor.fill_(0xA5)
        for index, obj in enumerate(objects):
            assert torch.equal(
                obj.raw_tensor, torch.arange(slot, dtype=torch.uint8) + index
            )
        del obj
        assert torch.all(reused[0].raw_tensor == 0xA5)
        manager.free(objects + reused)
        objects.clear()
        reused.clear()
        assert manager.get_memory_usage()[0] == 0
        with pytest.raises(L1ReconfigureError, match="already|duplicate"):
            manager.add_device(path, slot)
        with pytest.raises(L1ReconfigureError):
            manager.add_device(str(tmp_path / "missing"), slot)
        assert len(manager.get_arena_statuses()) == 1
    finally:
        manager.close()
    assert _open_fd_count(path) == 0
