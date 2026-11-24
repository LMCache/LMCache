# SPDX-License-Identifier: Apache-2.0
"""Online pressure-driven pooling through real HTTP and private DAX mappings.

Run with RUN_DEVDAX_L1_INTEGRATION=1. Only the physical provider is a test double;
the worker, hotplug routes, allocator, and payload accesses use production code.
"""

# Standard
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import os
import socket
import threading

# Third Party
from fastapi import FastAPI
import httpx
import pytest
import torch
import uvicorn

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_orchestrator.pooling import (
    MemoryAllocation,
    MemoryPoolController,
)
from lmcache.v1.multiprocess.http_apis.l1_reconfigure_api import router
from lmcache.v1.multiprocess.http_reconfigure import ReconfigureInstanceMiddleware
from tests.v1.distributed.test_dax_l2_adapter import (
    create_object_key,
    wait_for_condition,
)

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("RUN_DEVDAX_L1_INTEGRATION") != "1",
        reason="set RUN_DEVDAX_L1_INTEGRATION=1",
    ),
]

SLOT_BYTES = 4096


@dataclass
class FilePool:
    """Provision a real file and verify unmapping before physical return."""

    path: Path
    acquired: int = 0
    released: int = 0

    def acquire(
        self, consumer_id: str, request_id: str, size_bytes: int
    ) -> MemoryAllocation:
        """Create the exclusive backing range requested by the pressure policy."""
        assert consumer_id == "pressure-worker" and request_id.endswith("/burst")
        assert self.acquired == 0 and size_bytes == 4 * SLOT_BYTES
        self.path.write_bytes(bytes(size_bytes))
        self.acquired += 1
        return MemoryAllocation("burst-lease", str(self.path), 0, size_bytes)

    def release(self, allocation_id: str) -> None:
        """Return the lease only after the worker actually unmapped its file."""
        assert allocation_id == "burst-lease" and self.released == 0
        assert str(self.path) not in Path("/proc/self/maps").read_text()
        self.path.unlink()
        self.released += 1


@dataclass
class Traffic:
    """Only measured outcomes cross from the workload into the policy loop."""

    failures: int = 0
    reads: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> tuple[int, int]:
        """Return allocation failures and completed anchor reads atomically."""
        with self.lock:
            return self.failures, self.reads


@contextmanager
def http_worker(
    manager: StorageManager, stopped: threading.Event
) -> Iterator[httpx.Client]:
    """Serve production hotplug routes on an isolated loopback socket."""
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(ReconfigureInstanceMiddleware)
    app.state.engine = SimpleNamespace(storage_manager=manager)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            assert wait_for_condition(lambda: server.started)
            with httpx.Client(
                base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
                timeout=5,
                trust_env=False,
            ) as client:
                yield client
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()
            stopped.set()


def read_checked(l1: L1ManagerInterface, key: ObjectKey, value: int) -> None:
    """Validate every byte under a real read reservation and release its lock."""
    error, obj = l1.reserve_read([key])[key]
    assert error is L1Error.SUCCESS and obj is not None
    try:
        assert obj.raw_tensor is not None
        assert torch.all(obj.raw_tensor == value)
    finally:
        l1.finish_read([key])


def workload(
    manager: StorageManager,
    traffic: Traffic,
    stop: threading.Event,
    sampled_peak: threading.Event,
) -> None:
    """Ramp a working set to six slots, retire the burst, and keep serving reads."""
    layout = MemoryLayoutDesc(shapes=[torch.Size([SLOT_BYTES])], dtypes=[torch.uint8])
    keys = [create_object_key(i, model_name="pool-pressure") for i in range(6)]
    l1 = manager.l1_managers()[0]
    resident = full_rounds = 0
    retired = False
    while not stop.is_set():
        if not retired and resident < len(keys):
            key = keys[resident]
            objects = manager.reserve_write([key], layout)
            if key in objects:
                obj = objects[key]
                assert obj.raw_tensor is not None
                obj.raw_tensor.fill_(resident + 17)
                manager.finish_write([key], l1_owners_of(objects))
                del obj, objects
                resident += 1
            else:
                with traffic.lock:
                    traffic.failures += 1
        for index in range(resident):
            read_checked(l1, keys[index], index + 17)
        if resident:
            with traffic.lock:
                traffic.reads += 1
        if resident == len(keys):
            full_rounds += 1
            # Keep the peak observable even when CI schedules the policy slowly.
            if full_rounds >= 32 and sampled_peak.is_set():
                assert manager.delete_l1_keys(keys[2:]) == (4, 0)
                resident, retired = 2, True
        stop.wait(0.005)


def test_online_capacity_tracks_workload_pressure(tmp_path: Path) -> None:
    """Saturation drives growth; sustained lower occupancy returns capacity online."""
    primary = tmp_path / "primary"
    primary.write_bytes(bytes(2 * SLOT_BYTES))
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
    pool, traffic, stop = FilePool(tmp_path / "pooled"), Traffic(), threading.Event()
    sampled_peak, http_stopped = threading.Event(), threading.Event()
    future: Future[None] = Future()

    def run_workload() -> None:
        try:
            workload(manager, traffic, stop, sampled_peak)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(None)

    worker = threading.Thread(target=run_workload, daemon=True)
    try:
        with http_worker(manager, http_stopped) as client:
            controller = MemoryPoolController(pool, client, "pressure-worker", "l1")
            worker.start()
            low_samples = peak_used = 0
            grow_reads = shrink_reads = None
            capacities: list[int] = []
            deadline = monotonic() + 15
            try:
                while monotonic() < deadline:
                    if future.done():
                        future.result()
                        pytest.fail("workload stopped before the pressure policy")
                    response = client.get("/reconfigure/dax/l1/status")
                    response.raise_for_status()
                    arenas = response.json()["arenas"]
                    used = sum(arena["used_bytes"] for arena in arenas)
                    capacity = sum(arena["size_in_bytes"] for arena in arenas)
                    if not capacities or capacities[-1] != capacity:
                        capacities.append(capacity)
                    failures, reads = traffic.snapshot()
                    peak_used = max(peak_used, used)
                    if used == 6 * SLOT_BYTES:
                        sampled_peak.set()
                    if grow_reads is None and used == capacity and failures:
                        assert capacity == 2 * SLOT_BYTES and reads > 0
                        grow_reads = reads
                        controller.grow("burst", 4 * SLOT_BYTES)
                    elif grow_reads is not None and shrink_reads is None:
                        low_samples = low_samples + 1 if used <= 2 * SLOT_BYTES else 0
                        if peak_used == 6 * SLOT_BYTES and low_samples >= 3:
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
                assert pool.acquired == pool.released == 1
                assert grow_reads is not None and shrink_reads is not None
                assert traffic.snapshot()[1] >= shrink_reads + 10
            finally:
                stop.set()
                worker.join(timeout=5)
                assert not worker.is_alive()
                future.result()
    finally:
        # A stuck daemon may own backend locks; retain its fixture until exit.
        if http_stopped.is_set() and not worker.is_alive():
            manager.close()
