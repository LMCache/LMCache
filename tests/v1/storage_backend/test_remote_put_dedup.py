# SPDX-License-Identifier: Apache-2.0
"""Concurrent stores of the same key must be deduplicated atomically.

Regression test for https://github.com/LMCache/LMCache/issues/5914: the
duplicate-store check and the "store pending" marker used to be two separate
lock holds, so two concurrent callers could both pass the check and both submit.
"""

# Standard
from typing import List
from unittest import mock
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class CountingConnector(RemoteConnector):
    """Records every put so the test can count submissions."""

    def __init__(self):
        self.storage = {}
        self.put_count = 0
        self.release = threading.Event()

    async def exists(self, key):
        return key in self.storage

    def exists_sync(self, key):
        return key in self.storage

    async def put(self, key, value):
        self.put_count += 1
        # Hold the first write open so the second caller is still racing it.
        await asyncio.get_running_loop().run_in_executor(None, self.release.wait, 5.0)
        self.storage[key] = value

    async def get(self, key):
        return self.storage.get(key)

    async def close(self):
        pass

    async def list(self) -> List[str]:
        return []


@mock.patch.object(torch_dev, "Stream")
def test_concurrent_put_of_same_key_submits_once(mock_stream):
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=5.0,
        local_disk=None,
        max_local_disk_size=0.0,
        remote_url="mock://100",
        remote_serde="naive",
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
    )
    metadata = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=False,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
    )

    allocator = AdHocMemoryAllocator()
    local_cpu_backend = LocalCPUBackend(config, metadata, memory_allocator=allocator)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    connector = CountingConnector()
    backend.connection = connector

    # Force the two callers to interleave in the window between the duplicate
    # check and the insert into put_tasks. On the unfixed code both callers see
    # an empty set here and both go on to submit. Once the check and the insert
    # happen under one lock hold the second caller never reaches this hook: it
    # blocks on the lock, then finds the key already pending.
    barrier = threading.Barrier(2, timeout=2.0)
    original_exists = backend.exists_in_put_tasks

    def exists_then_wait(key):
        result = original_exists(key)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return result

    backend.exists_in_put_tasks = exists_then_wait

    key = CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash="dedup_hash",
        dtype=torch.float32,
    )

    def store():
        memory_obj = local_cpu_backend.allocate(
            torch.Size([2, 16, 8, 128]), torch.bfloat16, MemoryFormat.KV_T2D
        )
        assert memory_obj is not None
        backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()

    threads = [threading.Thread(target=store) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # `submit_put_task` only schedules the put on the event loop, so joining the
    # threads says nothing about whether the connector has been called yet. Wait
    # for the put to START: it increments the count and then blocks on `release`,
    # so the marker below is still pending. Reading the count straight after the
    # join raced the event loop and saw 0 on the slower CI runners.
    started = threading.Event()
    for _ in range(100):
        if connector.put_count >= 1:
            break
        started.wait(0.05)

    # Exactly one store is in flight, and the key is still marked pending.
    assert connector.put_count == 1
    assert original_exists(key)

    # Letting it finish clears the marker.
    connector.release.set()
    deadline = threading.Event()
    for _ in range(50):
        if not original_exists(key):
            break
        deadline.wait(0.1)
    assert not original_exists(key)

    loop.call_soon_threadsafe(loop.stop)
