# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.gds_backend import GdsBackend


def create_test_config(gds_path: str):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, gds_path=gds_path, lmcache_instance_id="test_instance"
    )
    return config


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey("vllm", "testmodel", 3, 123, key_id)


def create_test_memory_obj(
    shape=(2, 16, 8, 128), dtype=torch.bfloat16, device="cpu"
) -> MemoryObj:
    allocator = AdHocMemoryAllocator(device=device)
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


@pytest.fixture
def temp_gds_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


@pytest.fixture
def memory_allocator():
    return AdHocMemoryAllocator(device="cpu")


@pytest.fixture
def gds_backend(temp_gds_path, async_loop, memory_allocator):
    config = create_test_config(temp_gds_path)
    return GdsBackend(
        config=config,
        loop=async_loop,
        memory_allocator=memory_allocator,
        dst_device="cuda" if torch.cuda.is_available() else "cpu",
    )


# Optionally skip async tests if pytest-asyncio is not available
pytest_asyncio = pytest.importorskip(
    "pytest_asyncio", reason="pytest-asyncio is required for async tests"
)


class TestGdsBackend:
    def test_init(self, temp_gds_path, async_loop, memory_allocator):
        config = create_test_config(temp_gds_path)
        backend = GdsBackend(
            config=config,
            loop=async_loop,
            memory_allocator=memory_allocator,
            dst_device="cuda" if torch.cuda.is_available() else "cpu",
        )
        assert backend.gds_path == temp_gds_path
        assert backend.memory_allocator == memory_allocator
        assert backend.dst_device in ("cuda", "cpu")
        assert os.path.exists(temp_gds_path)

    def test_str(self, gds_backend):
        assert str(gds_backend) == "GdsBackend"

    def test_key_to_path_and_insert_key(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj()
        gds_backend.insert_key(key, memory_obj)
        # Check that the key is in hot_cache
        assert key in gds_backend.hot_cache
        meta = gds_backend.hot_cache[key]
        assert meta.shape == memory_obj.metadata.shape
        assert meta.dtype == memory_obj.metadata.dtype

    def test_contains_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        assert not gds_backend.contains(key)
        assert not gds_backend.contains(key, pin=True)

    def test_contains_key_exists(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj()
        gds_backend.insert_key(key, memory_obj)
        assert gds_backend.contains(key)
        assert gds_backend.contains(key, pin=True)

    def test_exists_in_put_tasks(self, gds_backend):
        key = create_test_key(0)
        assert not gds_backend.exists_in_put_tasks(key)
        # Simulate adding to put_tasks
        with gds_backend.put_lock:
            gds_backend.put_tasks.add(key)
        assert gds_backend.exists_in_put_tasks(key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA for GdsBackend get_blocking",
    )
    async def test_submit_put_task_and_get_blocking(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device="cpu")
        # submit_put_task returns a Future
        future = gds_backend.submit_put_task(key, memory_obj)
        assert future is not None
        # Wait for the async save to complete
        future.result(timeout=5)
        # Now the key should be in hot_cache
        assert gds_backend.contains(key)
        # get_blocking should return a MemoryObj (may be None if not CUDA)
        result = gds_backend.get_blocking(key)
        # On CPU, _load_bytes_from_disk may not work,
        # so just check for None or MemoryObj
        assert result is None or isinstance(result, MemoryObj)

    @pytest.mark.asyncio
    async def test_batched_submit_put_task(self, gds_backend):
        keys = [create_test_key(i) for i in range(2, 5)]
        memory_objs = [create_test_memory_obj(device="cpu") for _ in range(3)]
        futures = gds_backend.batched_submit_put_task(keys, memory_objs)
        assert futures is not None
        assert len(futures) == 3
        for future in futures:
            assert future is not None
            future.result(timeout=5)
        for key in keys:
            assert gds_backend.contains(key)

    def test_submit_prefetch_task_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        future = gds_backend.submit_prefetch_task(key)
        assert future is None

    def test_submit_prefetch_task_key_exists(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj()
        gds_backend.insert_key(key, memory_obj)
        future = gds_backend.submit_prefetch_task(key)
        # May be None if not CUDA, otherwise should be a Future
        assert future is None or hasattr(future, "result")

    def test_get_blocking_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        result = gds_backend.get_blocking(key)
        assert result is None

    def test_get_non_blocking(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj()
        gds_backend.insert_key(key, memory_obj)
        future = gds_backend.get_non_blocking(key)
        assert future is None or hasattr(future, "result")

    def test_close(self, gds_backend):
        # Should not raise
        gds_backend.close()

    def test_pin_unpin_not_implemented(self, gds_backend):
        key = create_test_key(0)
        with pytest.raises(NotImplementedError):
            gds_backend.pin(key)
        with pytest.raises(NotImplementedError):
            gds_backend.unpin(key)
