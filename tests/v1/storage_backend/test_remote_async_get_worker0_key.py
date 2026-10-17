# SPDX-License-Identifier: Apache-2.0
"""In MLA worker-id-as-0 mode the chunks are stored under worker 0.
batched_get_non_blocking must read the same keys as batched_async_contains."""

# Standard
from types import SimpleNamespace
import asyncio

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class _Conn:
    def __init__(self):
        self.contains_keys = None
        self.get_keys = None

    def support_batched_async_contains(self):
        return True

    async def batched_async_contains(self, lookup_id, keys, pin=False):
        self.contains_keys = keys
        return len(keys)

    async def batched_get_non_blocking(self, lookup_id, keys):
        self.get_keys = keys
        return ["obj"] * len(keys)


def _backend(as0: bool) -> RemoteBackend:
    backend = object.__new__(RemoteBackend)
    backend.connection = _Conn()  # type: ignore[assignment]
    backend.local_cpu_backend = object()  # type: ignore[assignment]
    backend.config = SimpleNamespace(blocking_timeout_secs=5)  # type: ignore[assignment]
    backend._mla_worker_id_as0_mode = as0
    return backend


def test_async_get_reads_the_keys_contains_found():
    keys = [CacheEngineKey("m", 8, 3, h, torch.bfloat16) for h in (11, 22)]
    backend = _backend(as0=True)
    asyncio.run(backend.batched_async_contains("req", keys))
    out = asyncio.run(backend.batched_get_non_blocking("req", keys))
    assert [k.worker_id for k in backend.connection.contains_keys] == [0, 0]
    assert [k.worker_id for k in backend.connection.get_keys] == [0, 0]
    assert [k.chunk_hash for k in backend.connection.get_keys] == [11, 22]
    assert out == ["obj", "obj"]


def test_keys_unchanged_outside_as0_mode():
    keys = [CacheEngineKey("m", 8, 3, 11, torch.bfloat16)]
    backend = _backend(as0=False)
    asyncio.run(backend.batched_get_non_blocking("req", keys))
    assert backend.connection.get_keys[0].worker_id == 3
