# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Redis non-blocking batch-get prefix semantics.

The Redis transports below are deliberately limited to the redis-py
construction and GET boundaries.  Each test creates a real LMCache connector
and reaches its public ``batched_get_non_blocking`` API, so metadata decoding,
CPU allocation, byte copying, and the priority executor stay on the path
under test.
"""

# Standard
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import (
    TensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, TensorMemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import redis_connector
from lmcache.v1.storage_backend.connector.redis_connector import (
    RedisClusterConnector,
    RedisConnector,
)

# Local
from ..utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    dumb_metadata,
    init_asyncio_loop,
)

ConnectorKind = Literal["redis", "redis-cluster"]
ReadResponse = bytes | Exception | None
RedisConnectorUnderTest = RedisConnector | RedisClusterConnector

_PAYLOADS = (b"payload0", b"payload1", b"payload2")
_REMOTE_SHAPE = torch.Size([2, 2, 2])


@dataclass(frozen=True)
class ReadCase:
    """One scripted Redis read outcome and its consecutive-prefix result."""

    name: str
    outcomes: tuple[Literal["hit", "missing", "error"], ...]
    prefix_indexes: tuple[int, ...]


_READ_CASES = (
    ReadCase("all_hits", ("hit", "hit", "hit"), (0, 1, 2)),
    ReadCase("middle_metadata_missing", ("hit", "missing", "hit"), (0,)),
    ReadCase("first_metadata_missing", ("missing", "hit", "hit"), ()),
    ReadCase("middle_metadata_get_error", ("hit", "error", "hit"), (0,)),
)


class RecordingTensorMemoryAllocator(TensorMemoryAllocator):
    """Small real tensor allocator that records each successful free once."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__(tensor)
        self.allocated_memory_objs: list[TensorMemoryObj] = []
        self.freed_memory_obj_ids: list[int] = []
        self.freed_payloads: dict[int, list[bytes]] = {}

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[TensorMemoryObj]:
        """Allocate with the production allocator and retain the test handle."""
        memory_obj = super().allocate(shapes, dtypes, fmt, allocator_type)
        if memory_obj is not None:
            self.allocated_memory_objs.append(memory_obj)
        return memory_obj

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Record every free attempt before delegating to production logic."""
        memory_obj_id = id(memory_obj)
        self.freed_memory_obj_ids.append(memory_obj_id)
        self.freed_payloads.setdefault(memory_obj_id, []).append(
            bytes(memory_obj.byte_array)
        )
        super().free(memory_obj, allocator_type)


class ScriptedRedisClient:
    """Async redis-py I/O boundary fake keyed by the real connector key strings."""

    def __init__(self, responses: dict[str, ReadResponse]) -> None:
        self.responses = responses
        self.get_calls: list[str] = []

    async def get(self, key: str) -> bytes | None:
        """Return the scripted value or surface the scripted redis-py error."""
        self.get_calls.append(key)
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response


def _remote_metadata_bytes(payload: bytes) -> bytes:
    """Serialize the real metadata consumed by the connector's ``_get`` path."""
    return RemoteMetadata(
        length=len(payload),
        shapes=[_REMOTE_SHAPE],
        dtypes=[torch.uint8],
        fmt=MemoryFormat.KV_2TD,
    ).serialize()


def _scripted_responses(
    keys: list[CacheEngineKey], case: ReadCase
) -> dict[str, ReadResponse]:
    """Build metadata/KV responses that exercise each real per-key fetch."""
    responses: dict[str, ReadResponse] = {}
    for key, payload, outcome in zip(keys, _PAYLOADS, case.outcomes, strict=True):
        key_string = key.to_string()
        if outcome == "hit":
            responses[key_string + "metadata"] = _remote_metadata_bytes(payload)
            responses[key_string + "kv_bytes"] = payload
        elif outcome == "missing":
            responses[key_string + "metadata"] = None
        else:
            responses[key_string + "metadata"] = RuntimeError(
                "scripted Redis GET error"
            )
    return responses


def _create_local_backend(
    allocator: RecordingTensorMemoryAllocator,
) -> LocalCPUBackend:
    """Create a LocalCPUBackend over a 128 KiB CPU-only real tensor allocator."""
    return LocalCPUBackend(
        config=LMCacheEngineConfig.from_defaults(),
        metadata=dumb_metadata(),
        memory_allocator=allocator,
    )


def _connection_pool_factory(pool: object) -> Callable[..., object]:
    """Return the redis-py ConnectionPool factory used at construction time."""

    def from_url(_url: str, **_kwargs: object) -> object:
        """Return the per-test inert pool without opening a socket."""
        return pool

    return from_url


def _client_factory(client: ScriptedRedisClient) -> Callable[..., ScriptedRedisClient]:
    """Return the redis-py client factory used at construction time."""

    def from_pool(_pool: object, **_kwargs: object) -> ScriptedRedisClient:
        """Return the per-test scripted async client."""
        return client

    return from_pool


def _cluster_factory(client: ScriptedRedisClient) -> Callable[..., ScriptedRedisClient]:
    """Return the RedisCluster construction boundary for one scripted client."""

    def create_cluster(**_kwargs: object) -> ScriptedRedisClient:
        """Return the per-test scripted async cluster client."""
        return client

    return create_cluster


def _create_connector(
    connector_kind: ConnectorKind,
    loop: asyncio.AbstractEventLoop,
    local_backend: LocalCPUBackend,
    client: ScriptedRedisClient,
    monkeypatch: pytest.MonkeyPatch,
) -> RedisConnectorUnderTest:
    """Construct the selected real connector with only redis-py boundaries faked."""
    if connector_kind == "redis":
        pool = object()
        monkeypatch.setattr(
            redis_connector.redis.ConnectionPool,
            "from_url",
            _connection_pool_factory(pool),
        )
        monkeypatch.setattr(
            redis_connector.redis.Redis,
            "from_pool",
            _client_factory(client),
        )
        return RedisConnector("redis://test.invalid:1", loop, local_backend)

    monkeypatch.setattr(
        redis_connector,
        "RedisCluster",
        _cluster_factory(client),
    )
    return RedisClusterConnector(
        hosts_and_ports=[("127.0.0.1", 1)],
        username="",
        password="",
        loop=loop,
        local_cpu_backend=local_backend,
    )


def _memory_obj_bytes(memory_obj: MemoryObj) -> bytes:
    """Read the public byte view used by the remote connector copy path."""
    return bytes(memory_obj.byte_array)


def _shutdown_executor(
    connector: RedisConnectorUnderTest,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Stop the public queue-executor API without calling connector.close()."""
    shutdown = asyncio.run_coroutine_threadsafe(
        connector.pq_executor.shutdown_async(wait=False), loop
    )
    shutdown.result(timeout=2)


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("connector_kind", ["redis", "redis-cluster"])
@pytest.mark.parametrize("case", _READ_CASES, ids=[case.name for case in _READ_CASES])
def test_redis_batched_get_non_blocking_keeps_consecutive_prefix(
    connector_kind: ConnectorKind,
    case: ReadCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep only the successful prefix and release every later successful read.

    The fixture deliberately simulates a contains/get time-of-check-to-time-of-use
    race: a metadata GET can miss or error between a prior contains operation
    and this public batch GET.  A later hit may not be compacted into that
    missing key's position.
    """
    async_loop, async_thread = init_asyncio_loop()
    allocator = RecordingTensorMemoryAllocator(
        torch.empty(128 * 1024, dtype=torch.uint8)
    )
    local_backend = _create_local_backend(allocator)
    connector: Optional[RedisConnectorUnderTest] = None

    try:
        keys = [dumb_cache_engine_key(index) for index in range(len(_PAYLOADS))]
        client = ScriptedRedisClient({})
        connector = _create_connector(
            connector_kind,
            async_loop,
            local_backend,
            client,
            monkeypatch,
        )
        client.responses.update(_scripted_responses(keys, case))

        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get_non_blocking("prefix-regression", keys),
            async_loop,
        )
        prefix = future.result(timeout=2)

        expected_prefix = [_PAYLOADS[index] for index in case.prefix_indexes]
        assert [
            _memory_obj_bytes(memory_obj) for memory_obj in prefix
        ] == expected_prefix

        expected_get_calls = {key.to_string() + "metadata" for key in keys} | {
            key.to_string() + "kv_bytes"
            for key, outcome in zip(keys, case.outcomes, strict=True)
            if outcome == "hit"
        }
        assert Counter(client.get_calls) == Counter(expected_get_calls)

        returned_ids = {id(memory_obj) for memory_obj in prefix}
        tail_memory_objs = [
            memory_obj
            for memory_obj in allocator.allocated_memory_objs
            if id(memory_obj) not in returned_ids
        ]
        expected_tail_payloads = [
            payload
            for index, payload in enumerate(_PAYLOADS)
            if index not in case.prefix_indexes and case.outcomes[index] == "hit"
        ]
        assert all(not memory_obj.is_valid() for memory_obj in tail_memory_objs)
        assert Counter(allocator.freed_memory_obj_ids) == Counter(
            id(memory_obj) for memory_obj in tail_memory_objs
        )
        assert sorted(
            allocator.freed_payloads[id(memory_obj)][0]
            for memory_obj in tail_memory_objs
        ) == sorted(expected_tail_payloads)

        for memory_obj in prefix:
            memory_obj.ref_count_down()
        assert all(
            not memory_obj.is_valid() for memory_obj in allocator.allocated_memory_objs
        )
        assert Counter(allocator.freed_memory_obj_ids) == Counter(
            id(memory_obj) for memory_obj in allocator.allocated_memory_objs
        )
    finally:
        for memory_obj in allocator.allocated_memory_objs:
            if memory_obj.is_valid():
                memory_obj.ref_count_down()
        if connector is not None:
            _shutdown_executor(connector, async_loop)
        close_asyncio_loop(async_loop, async_thread)
