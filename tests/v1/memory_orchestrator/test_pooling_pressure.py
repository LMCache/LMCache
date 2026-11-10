# SPDX-License-Identifier: Apache-2.0
"""Online pressure-driven pooling through ASGI routes and file-backed DAX.

Run with RUN_DEVDAX_L1_INTEGRATION=1. Only the physical provider is mocked.
"""

# Standard
from concurrent.futures import Future
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
import os
import threading

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_orchestrator.pooling import (
    MemoryAllocation,
    MemoryPoolController,
    MemoryPoolProvider,
)
from lmcache.v1.multiprocess.http_apis.l1_reconfigure_api import router
from lmcache.v1.multiprocess.http_reconfigure import ReconfigureInstanceMiddleware
from tests.v1.distributed.test_dax_l2_adapter import create_object_key

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("RUN_DEVDAX_L1_INTEGRATION") != "1",
        reason="set RUN_DEVDAX_L1_INTEGRATION=1",
    ),
]
SLOT_BYTES = 4096


def test_online_capacity_tracks_workload_pressure(tmp_path: Path) -> None:
    """Measured saturation grows capacity; retiring the burst shrinks it online."""
    primary, pooled = tmp_path / "primary", tmp_path / "pooled"
    primary.write_bytes(bytes(2 * SLOT_BYTES))
    pooled.write_bytes(bytes(4 * SLOT_BYTES))
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=2 * SLOT_BYTES,
                    devdax_path=str(primary),
                    use_lazy=False,
                    shm_name="",
                    align_bytes=SLOT_BYTES,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
        )
    )
    allocation = MemoryAllocation("burst-lease", str(pooled), 0, 4 * SLOT_BYTES)
    pool = Mock(spec=MemoryPoolProvider)
    pool.acquire.return_value = allocation

    def release(allocation_id: str) -> None:
        assert allocation_id == allocation.allocation_id
        assert str(pooled) not in Path("/proc/self/maps").read_text()
        pooled.unlink()

    pool.release.side_effect = release
    stop, sampled_peak = threading.Event(), threading.Event()
    lock = threading.Lock()
    traffic = {"failures": 0, "reads": 0}
    result: Future[None] = Future()

    def workload() -> None:
        """Fill six slots, retire burst keys after observation, and keep reading."""
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([SLOT_BYTES])], dtypes=[torch.uint8]
        )
        keys = [create_object_key(i, model_name="pool-pressure") for i in range(6)]
        l1 = manager.l1_managers()[0]
        resident, retired = 0, False
        try:
            while not stop.is_set():
                if not retired and resident < len(keys):
                    key = keys[resident]
                    objects = manager.reserve_write([key], layout)
                    if key in objects:
                        tensor = objects[key].raw_tensor
                        assert tensor is not None
                        tensor.fill_(resident + 17)
                        manager.finish_write([key], l1_owners_of(objects))
                        del tensor, objects
                        resident += 1
                    else:
                        with lock:
                            traffic["failures"] += 1
                for index in range(resident):
                    key = keys[index]
                    error, obj = l1.reserve_read([key])[key]
                    assert error is L1Error.SUCCESS and obj is not None
                    try:
                        assert obj.raw_tensor is not None
                        assert torch.all(obj.raw_tensor == index + 17)
                    finally:
                        l1.finish_read([key])
                    del obj
                if resident:
                    with lock:
                        traffic["reads"] += 1
                # The handshake keeps the pressure peak observable on slow CI.
                if resident == len(keys) and sampled_peak.is_set():
                    assert manager.delete_l1_keys(keys[2:]) == (4, 0)
                    resident, retired = 2, True
                stop.wait(0.005)
        except BaseException as error:
            result.set_exception(error)
        else:
            result.set_result(None)

    worker = threading.Thread(target=workload, daemon=True)
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(ReconfigureInstanceMiddleware)
    app.state.engine = SimpleNamespace(storage_manager=manager)
    try:
        with TestClient(app) as client:
            # TestClient may use httpx2 internally, with the same request interface.
            controller = MemoryPoolController(
                pool, cast(httpx.Client, client), "pressure-worker", "l1"
            )
            worker.start()
            low_samples = 0
            grow_reads = shrink_reads = None
            capacities: list[int] = []
            deadline = monotonic() + 15
            try:
                while monotonic() < deadline:
                    if result.done():
                        result.result()
                        pytest.fail("workload stopped before the pressure policy")
                    response = client.get("/reconfigure/dax/l1/status")
                    response.raise_for_status()
                    arenas = response.json()["arenas"]
                    used = sum(arena["used_bytes"] for arena in arenas)
                    capacity = sum(arena["size_in_bytes"] for arena in arenas)
                    if not capacities or capacities[-1] != capacity:
                        capacities.append(capacity)
                    with lock:
                        failures, reads = traffic["failures"], traffic["reads"]
                    if used == 6 * SLOT_BYTES:
                        sampled_peak.set()
                    if grow_reads is None and used == capacity and failures:
                        assert reads > 0
                        grow_reads = reads
                        controller.grow("burst", 4 * SLOT_BYTES)
                    elif grow_reads is not None and shrink_reads is None:
                        low_samples = low_samples + 1 if used <= 2 * SLOT_BYTES else 0
                        if sampled_peak.is_set() and low_samples >= 3:
                            assert reads > grow_reads
                            assert controller.shrink("burst")
                            shrink_reads = reads
                    elif shrink_reads is not None and reads >= shrink_reads + 10:
                        break
                    stop.wait(0.01)
                else:
                    pytest.fail(
                        "online pressure policy did not grow and shrink in time"
                    )
                assert capacities == [2 * SLOT_BYTES, 6 * SLOT_BYTES, 2 * SLOT_BYTES]
                pool.acquire.assert_called_once()
                pool.release.assert_called_once_with(allocation.allocation_id)
            finally:
                stop.set()
                worker.join(timeout=5)
                assert not worker.is_alive()
                result.result()
    finally:
        # A stuck workload may own backend locks; retain its fixture until exit.
        if not worker.is_alive():
            manager.close()
