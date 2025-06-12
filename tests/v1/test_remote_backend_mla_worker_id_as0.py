# Standard
from typing import List

# Add import for mock
from unittest import mock
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class MockConnector(RemoteConnector):
    def __init__(self):
        self.storage = {}

    async def exists(self, key):
        return key in self.storage

    async def put(self, key, value):
        self.storage[key] = value

    async def get(self, key):
        return self.storage.get(key)

    async def close(self):
        pass

    async def list(self) -> List[str]:
        return []


# Mock the entire torch.cuda.Stream class
@mock.patch("torch.cuda.Stream")
def test_remote_backend_methods(mock_stream):
    # Create configuration
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=5.0,
        local_disk=None,
        max_local_disk_size=0.0,
        remote_url="lm://localhost:65432",
        remote_serde="naive",
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
        blend_recompute_ratio=0.15,
        blend_min_tokens=256,
        extra_config={"remote_enable_mla_worker_id_as0": True},
    )

    metadata = LMCacheEngineMetadata(
        model_name="test-model",
        fmt="vllm",
        kv_dtype=torch.float16,
        kv_shape=(2, 32, 1000, 1024, 1),
        use_mla=True,
        world_size=4,
        worker_id=2,
    )
    metadata0 = LMCacheEngineMetadata(
        model_name="test-model",
        fmt="vllm",
        kv_dtype=torch.float16,
        kv_shape=(1, 32, 1, 1024, 1),
        use_mla=True,
        world_size=4,
        worker_id=0,
    )

    # Create memory allocator and local backend
    # First Party
    from lmcache.v1.memory_management import AdHocMemoryAllocator

    pin_allocator = AdHocMemoryAllocator()
    local_cpu_backend = LocalCPUBackend(config, pin_allocator)

    loop = asyncio.new_event_loop()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
        lookup_server=None,
    )
    backend.connection = MockConnector()

    # Start the event loop in a separate thread
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # Create key
    key = CacheEngineKey(
        fmt="vllm",
        model_name="test-model",
        world_size=4,
        worker_id=2,
        chunk_hash="test_hash",
    )

    backend0 = RemoteBackend(
        config=config,
        metadata=metadata0,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
        lookup_server=None,
    )
    backend0.connection = backend.connection
    # Create key
    key0 = CacheEngineKey(
        fmt="vllm",
        model_name="test-model",
        world_size=4,
        worker_id=0,
        chunk_hash="test_hash",
    )

    # Test not contains before adding data
    assert not backend.contains(key)
    assert not backend0.contains(key0)

    # Test submit_put_task
    memory_obj = local_cpu_backend.allocate(torch.Size([10, 10]), torch.float32)
    future = backend.submit_put_task(key, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test not contains after adding data since worker_id 2 skipped put
    assert not backend.contains(key)

    future = backend0.submit_put_task(key0, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test contains after adding data since worker_id 0 should put
    assert backend0.contains(key0)
    # Test contains after adding data since we use worker_id 0 instead
    assert backend.contains(key)

    # Test get_blocking
    retrieved = backend.get_blocking(key)
    assert retrieved is not None
    assert retrieved.get_shape() == torch.Size([10, 10])

    # Cleanup
    # Cancel all running tasks
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    # Wait until all tasks are cancelled
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    # Stop the event loop
    loop.call_soon_threadsafe(loop.stop)
    # Wait for the loop thread to finish
    loop_thread.join(timeout=1.0)
    # Then close the loop
    loop.close()
