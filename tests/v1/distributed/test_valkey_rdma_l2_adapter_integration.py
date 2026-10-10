# SPDX-License-Identifier: Apache-2.0
"""
Integration test for the Valkey RDMA L2 adapter in MP mode.

Real transfers against a **real** valkey-server running the valkey-large-object
module, through the installed ``valkey-glide-sync`` with RDMA support.
Skipped unless ``LMCACHE_VALKEY_RDMA_SERVER=<host>:<port>`` names such a
server and the installed client can do RDMA on this machine.

``LMCACHE_VALKEY_RDMA_PROVIDER`` selects the fabric provider, ``tcp`` by
default, which pairs with the module's ``Emulated`` provider and needs no
hardware::

    valkey-server --port 6379 \\
        --loadmodule ./libvalkey_large_object.so \\
            operating-mode Dram fabric-provider Emulated fabric-interfaces lo

    LMCACHE_VALKEY_RDMA_SERVER=127.0.0.1:6379 pytest <thisfile> -v

Set ``LMCACHE_VALKEY_RDMA_PROVIDER=efa-direct`` on EFA hardware.
"""

# Standard
from typing import Any, Callable
import os
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.valkey_rdma_l2_adapter import (
    ValkeyRdmaL2Adapter,
    ValkeyRdmaL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

SERVER_ENV = "LMCACHE_VALKEY_RDMA_SERVER"
PROVIDER_ENV = "LMCACHE_VALKEY_RDMA_PROVIDER"

ARENA_BYTES = 1 << 20
ALIGN = 4096
CHUNK_BYTES = 64 << 10


def _rdma_usable() -> bool:
    """Whether the installed glide client, not a test fake, can do RDMA here."""
    try:
        # Third Party
        import glide_sync
    except ImportError:
        return False
    if getattr(glide_sync, "__spec__", None) is None:
        return False
    client = getattr(glide_sync, "GlideClient", None)
    return bool(
        client is not None
        and hasattr(client, "rdma_usable")
        and client.rdma_available()
        and client.rdma_usable()
    )


pytestmark = [
    pytest.mark.skipif(
        not os.getenv(SERVER_ENV),
        reason=f"set {SERVER_ENV}=<host>:<port> to a server running the "
        "valkey-large-object module",
    ),
    pytest.mark.skipif(
        not _rdma_usable(),
        reason="needs valkey-glide-sync built with RDMA and libfabric on this machine",
    ),
]


class _Arena:
    """A stand-in for the L1 arena: one tensor, chunks carved out of it."""

    def __init__(self, size: int = ARENA_BYTES) -> None:
        self.tensor = torch.zeros(size, dtype=torch.uint8)
        self.desc = L1MemoryDesc(
            ptr=self.tensor.data_ptr(), size=size, align_bytes=ALIGN
        )
        self._next = 0

    def alloc(self, nbytes: int = CHUNK_BYTES, fill: float = 1.0) -> TensorMemoryObj:
        offset = self._next
        phy_size = -(-nbytes // ALIGN) * ALIGN
        self._next += phy_size
        assert self._next <= self.desc.size, "arena exhausted"
        raw = self.tensor[offset : offset + nbytes].view(torch.float32)
        raw.fill_(fill)
        meta = MemoryObjMetadata(
            shape=torch.Size([nbytes // 4]),
            dtype=torch.float32,
            address=offset,
            phy_size=phy_size,
            fmt=MemoryFormat.KV_2LTD,
            ref_count=1,
        )
        return TensorMemoryObj(raw, meta, parent_allocator=None)

    def bytes_of(self, obj: TensorMemoryObj) -> bytes:
        start = obj.meta.address
        return bytes(self.tensor[start : start + obj.get_size()].tolist())


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=f"rdma-it-{os.getpid()}",
        kv_rank=0,
        cache_salt="",
    )


def _wait(fd: int, ready: Callable[[], Any], timeout: float = 30.0) -> Any:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        result = ready()
        if result is not None:
            return result
    raise AssertionError(f"no completion within {timeout}s")


def _store(adapter: ValkeyRdmaL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_store_task(keys, objs)
    return _wait(
        adapter.get_store_event_fd(),
        lambda: adapter.pop_completed_store_tasks().get(task_id),
    )


def _load(adapter: ValkeyRdmaL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_load_task(keys, objs)
    return _wait(
        adapter.get_load_event_fd(), lambda: adapter.query_load_result(task_id)
    )


def _lookup(adapter: ValkeyRdmaL2Adapter, keys: list) -> Any:
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    return _wait(
        adapter.get_lookup_and_lock_event_fd(),
        lambda: adapter.query_lookup_and_lock_result(task_id),
    )


@pytest.fixture(scope="module")
def target() -> tuple[str, str]:
    return os.environ[SERVER_ENV], os.getenv(PROVIDER_ENV, "tcp")


@pytest.fixture
def arena() -> _Arena:
    return _Arena()


@pytest.fixture
def adapter(target: tuple[str, str], arena: _Arena):
    startup_nodes, provider = target
    config = ValkeyRdmaL2AdapterConfig.from_dict(
        {
            "type": "valkey_rdma",
            "startup_nodes": startup_nodes,
            "num_workers": 2,
            "rdma_provider": provider,
            "connection_timeout": 10.0,
            "request_timeout": 10.0,
        }
    )
    adapter = create_l2_adapter_from_registry(config, arena.desc)
    assert isinstance(adapter, ValkeyRdmaL2Adapter)
    yield adapter
    adapter.close()


class TestRdmaTransfers:
    def test_chunks_round_trip_through_the_fabric(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        keys = [_key(1), _key(2), _key(3)]
        sources = [arena.alloc(fill=float(i + 1)) for i in range(3)]

        result = _store(adapter, keys, sources)
        assert result.is_successful()
        assert result.bytes_transferred() == 3 * CHUNK_BYTES

        found = _lookup(adapter, keys + [_key(404)])
        assert all(found.test(i) for i in range(3))
        assert not found.test(3)
        adapter.submit_unlock(keys)

        destinations = [arena.alloc(fill=0.0) for _ in range(3)]
        loaded = _load(adapter, keys, destinations)
        assert all(loaded.test(i) for i in range(3))
        for src, dst in zip(sources, destinations, strict=True):
            assert arena.bytes_of(dst) == arena.bytes_of(src)

        adapter.delete(keys)
        assert not _lookup(adapter, keys).test(0)

    def test_a_missing_key_leaves_the_destination_alone(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        dst = arena.alloc(fill=7.0)
        before = arena.bytes_of(dst)
        assert not _load(adapter, [_key(404)], [dst]).test(0)
        assert arena.bytes_of(dst) == before

    def test_a_value_larger_than_the_chunk_is_refused_before_any_write(
        self, adapter: ValkeyRdmaL2Adapter, arena: _Arena
    ):
        # A bigger value is refused rather than overrunning the next slab.
        key = _key(5)
        assert _store(
            adapter, [key], [arena.alloc(2 * CHUNK_BYTES, 5.0)]
        ).is_successful()
        dst = arena.alloc(fill=0.0)
        neighbour = arena.alloc(fill=9.0)
        before_neighbour = arena.bytes_of(neighbour)

        assert not _load(adapter, [key], [dst]).test(0)

        assert arena.bytes_of(dst) == bytes(CHUNK_BYTES)
        assert arena.bytes_of(neighbour) == before_neighbour
        adapter.delete([key])

    def test_ttl_is_applied(self, target: tuple[str, str], arena: _Arena):
        startup_nodes, provider = target
        config = ValkeyRdmaL2AdapterConfig.from_dict(
            {
                "type": "valkey_rdma",
                "startup_nodes": startup_nodes,
                "num_workers": 1,
                "rdma_provider": provider,
                "ttl_seconds": 1,
            }
        )
        adapter = create_l2_adapter_from_registry(config, arena.desc)
        try:
            key = _key(6)
            assert _store(adapter, [key], [arena.alloc()]).is_successful()
            assert _lookup(adapter, [key]).test(0)
            adapter.submit_unlock([key])
            time.sleep(2.5)
            assert not _lookup(adapter, [key]).test(0)
        finally:
            adapter.close()

    def test_status_reports_the_rdma_settings(
        self, adapter: ValkeyRdmaL2Adapter, target: tuple[str, str]
    ):
        status = adapter.report_status()
        assert status["is_healthy"]
        assert status["type"] == "valkey_rdma"
        assert status["rdma_provider"] == target[1]
        assert status["l1_arena_bytes"] == ARENA_BYTES
