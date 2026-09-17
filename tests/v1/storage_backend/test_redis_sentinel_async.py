# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Redis Sentinel's asynchronous child-client boundary."""

# Standard
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.redis_connector import RedisSentinelConnector

# Local
from ..utils import close_asyncio_loop, dumb_cache_engine_key, init_asyncio_loop

pytestmark = pytest.mark.no_shared_allocator


def _small_cpu_backend() -> LocalCPUBackend:
    """Create an isolated CPU backend with a small tensor-backed allocator.

    Returns:
        A backend whose 64 KiB CPU allocator is sufficient for the tiny
        regression objects while making reference-count changes observable.
    """
    config = LMCacheEngineConfig.from_defaults(chunk_size=2)
    metadata = LMCacheMetadata(
        model_name="redis-sentinel-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 2, 1, 1),
    )
    allocator = TensorMemoryAllocator(torch.zeros(64 * 1024, dtype=torch.uint8))
    return LocalCPUBackend(config=config, metadata=metadata, memory_allocator=allocator)


def _small_memory_obj(local_cpu_backend: LocalCPUBackend) -> MemoryObj:
    """Allocate a deterministic small object and retain a caller reference.

    Args:
        local_cpu_backend: Backend that owns the object's CPU allocation.

    Returns:
        A 16-byte object with one additional caller-held reference, so the
        instrumented connector's one release can be asserted safely.

    Raises:
        AssertionError: If the small test allocator cannot provide the object.
    """
    memory_obj = local_cpu_backend.allocate(torch.Size([1, 2, 2, 2]), torch.float16)
    assert memory_obj is not None
    memory_obj.raw_data.copy_(
        torch.arange(memory_obj.raw_data.numel(), dtype=torch.uint8)
    )
    memory_obj.ref_count_up()
    return memory_obj


def _sentinel_child(
    connector: InstrumentedRemoteConnector,
) -> RedisSentinelConnector:
    """Return the concrete Sentinel connector through its public wrapper API.

    Args:
        connector: Public connector returned by :func:`CreateConnector`.

    Returns:
        The Redis Sentinel connector whose public ``master`` and ``slave``
        client handles are the mocked Redis network boundary.
    """
    return cast(RedisSentinelConnector, connector.getWrappedConnector())


@pytest.fixture
def sentinel_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[
    tuple[InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend]
]:
    """Create a public Sentinel connector on a dedicated running event loop.

    Args:
        monkeypatch: Pytest environment helper used to supply Sentinel's
            required service name without affecting other tests.

    Yields:
        The public instrumented connector, its running loop, and its small
        CPU backend. Redis network calls are provided by the autouse async
        Sentinel mock only; connector construction remains public.
    """
    monkeypatch.setenv("REDIS_SERVICE_NAME", "mymaster")
    async_loop, async_thread = init_asyncio_loop()
    local_cpu_backend = _small_cpu_backend()
    connector = CreateConnector(
        "redis-sentinel://sentinel.test:26379", async_loop, local_cpu_backend
    )
    try:
        yield connector, async_loop, local_cpu_backend
    finally:
        asyncio.run_coroutine_threadsafe(connector.close(), async_loop).result(
            timeout=10
        )
        close_asyncio_loop(async_loop, async_thread)
        local_cpu_backend.close()


def test_async_sentinel_child_mock_smoke(
    sentinel_connector: tuple[
        InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend
    ],
) -> None:
    """Verify the async child-client mock itself provides an awaited I/O control.

    This is a network-boundary control only: it does not invoke any Sentinel
    connector operation, so it remains valid on the unfixed baseline.

    Args:
        sentinel_connector: Public connector, running event loop, and CPU
            backend supplied by the fixture.

    Returns:
        ``None`` after direct awaits prove the master/replica mock can set,
        get, check, and close values independently of connector behavior.
    """
    connector, async_loop, _ = sentinel_connector
    child = _sentinel_child(connector)
    control_key = "async-child-control"

    assert not asyncio.run_coroutine_threadsafe(
        child.slave.exists(control_key), async_loop
    ).result(timeout=10)
    assert asyncio.run_coroutine_threadsafe(
        child.master.set(control_key, b"control"), async_loop
    ).result(timeout=10)
    assert asyncio.run_coroutine_threadsafe(
        child.slave.exists(control_key), async_loop
    ).result(timeout=10)
    assert (
        asyncio.run_coroutine_threadsafe(
            child.slave.get(control_key), async_loop
        ).result(timeout=10)
        == b"control"
    )
    asyncio.run_coroutine_threadsafe(child.master.close(), async_loop).result(
        timeout=10
    )
    asyncio.run_coroutine_threadsafe(child.slave.close(), async_loop).result(timeout=10)


def test_sentinel_async_exists_and_sync_report_miss_and_hit(
    sentinel_connector: tuple[
        InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend
    ],
) -> None:
    """Verify asynchronous and synchronous existence reads agree on a key.

    Args:
        sentinel_connector: Public connector, running event loop, and CPU
            backend supplied by the fixture.

    Returns:
        ``None`` after both public existence forms report a miss then a hit,
        and the wrapper releases exactly one source-object reference.
    """
    connector, async_loop, local_cpu_backend = sentinel_connector
    key = dumb_cache_engine_key(101)

    assert not asyncio.run_coroutine_threadsafe(
        connector.exists(key), async_loop
    ).result(timeout=10)
    assert connector.exists_sync(key) is False

    memory_obj = _small_memory_obj(local_cpu_backend)
    try:
        asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        ).result(timeout=10)
        assert memory_obj.get_ref_count() == 1
        assert asyncio.run_coroutine_threadsafe(
            connector.exists(key), async_loop
        ).result(timeout=10)
        assert connector.exists_sync(key) is True
    finally:
        memory_obj.ref_count_down()


@pytest.mark.parametrize("failed_write_index", [1, 2], ids=["payload", "metadata"])
def test_sentinel_put_orders_writes_and_recovers_after_failure(
    sentinel_connector: tuple[
        InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend
    ],
    failed_write_index: int,
) -> None:
    """Verify each write failure leaves metadata unpublished and recoverable.

    Args:
        sentinel_connector: Public connector, running event loop, and CPU
            backend supplied by the fixture.
        failed_write_index: One-based write position that the async Redis
            boundary rejects: payload first or metadata second.

    Returns:
        ``None`` after assertions verify ordered awaited writes, absent
        metadata after failure, one wrapper-owned release, and a normal
        same-key put/get after restoring the child client.
    """
    connector, async_loop, local_cpu_backend = sentinel_connector
    key = dumb_cache_engine_key(102)
    key_str = key.to_string()
    child = _sentinel_child(connector)
    original_set = child.master.set
    set_mock = AsyncMock(
        side_effect=[
            (
                RuntimeError(f"write {write_index} rejected")
                if write_index == failed_write_index
                else True
            )
            for write_index in (1, 2)
        ]
    )
    child.master.set = set_mock
    memory_obj = _small_memory_obj(local_cpu_backend)

    try:
        with pytest.raises(RuntimeError, match=f"write {failed_write_index} rejected"):
            asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            ).result(timeout=10)

        assert [call.args[0] for call in set_mock.await_args_list] == [
            key_str + "kv_bytes",
            key_str + "metadata",
        ][:failed_write_index]
        assert key_str + "metadata" not in child.master.store
        assert memory_obj.get_ref_count() == 1

        child.master.set = original_set
        memory_obj.ref_count_up()
        asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        ).result(timeout=10)
        assert memory_obj.get_ref_count() == 1
        loaded = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result(timeout=10)
        assert loaded is not None
        try:
            assert bytes(loaded.byte_array) == bytes(memory_obj.byte_array)
        finally:
            loaded.ref_count_down()
    finally:
        child.master.set = original_set
        memory_obj.ref_count_down()


def test_sentinel_get_handles_miss_hit_and_stale_metadata(
    sentinel_connector: tuple[
        InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend
    ],
) -> None:
    """Verify public reads await misses, hits, and stale-metadata cleanup.

    Args:
        sentinel_connector: Public connector, running event loop, and CPU
            backend supplied by the fixture.

    Returns:
        ``None`` after assertions verify a miss, byte-preserving hit, and
        awaited deletion of metadata whose payload has disappeared.
    """
    connector, async_loop, local_cpu_backend = sentinel_connector
    key = dumb_cache_engine_key(103)
    key_str = key.to_string()
    child = _sentinel_child(connector)

    assert (
        asyncio.run_coroutine_threadsafe(connector.get(key), async_loop).result(
            timeout=10
        )
        is None
    )

    memory_obj = _small_memory_obj(local_cpu_backend)
    try:
        asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        ).result(timeout=10)
        loaded = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result(timeout=10)
        assert loaded is not None
        try:
            assert bytes(loaded.byte_array) == bytes(memory_obj.byte_array)
        finally:
            loaded.ref_count_down()

        asyncio.run_coroutine_threadsafe(
            child.master.delete(key_str + "kv_bytes"), async_loop
        ).result(timeout=10)
        delete_mock = AsyncMock(wraps=child.master.delete)
        child.master.delete = delete_mock

        assert (
            asyncio.run_coroutine_threadsafe(connector.get(key), async_loop).result(
                timeout=10
            )
            is None
        )
        delete_mock.assert_awaited_once_with(key_str + "metadata")
        assert key_str + "metadata" not in child.master.store
    finally:
        memory_obj.ref_count_down()


def test_sentinel_close_awaits_both_discovered_child_clients(
    sentinel_connector: tuple[
        InstrumentedRemoteConnector, asyncio.AbstractEventLoop, LocalCPUBackend
    ],
) -> None:
    """Verify public connector close awaits both Redis child-client closures.

    Args:
        sentinel_connector: Public connector, running event loop, and CPU
            backend supplied by the fixture.

    Returns:
        ``None`` after assertions verify both mocked child closes were awaited
        exactly once by the public close operation.
    """
    connector, async_loop, _ = sentinel_connector
    child = _sentinel_child(connector)
    master_close = AsyncMock(wraps=child.master.close)
    slave_close = AsyncMock(wraps=child.slave.close)
    child.master.close = master_close
    child.slave.close = slave_close

    asyncio.run_coroutine_threadsafe(connector.close(), async_loop).result(timeout=10)

    master_close.assert_awaited_once_with()
    slave_close.assert_awaited_once_with()
