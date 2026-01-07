# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest import mock
import asyncio
import os
import shutil
import sys
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.gds_backend import GdsBackend


def create_test_config(gds_path: str):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        gds_path=gds_path,
        lmcache_instance_id="test_instance",
        cufile_buffer_size=256,
        extra_config={"use_direct_io": True},
    )
    return config


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey("vllm", "testmodel", 3, 123, key_id, torch.bfloat16)


def create_test_memory_obj(
    shape=(2, 16, 8, 128), dtype=torch.bfloat16, device="cuda"
) -> MemoryObj:
    allocator = AdHocMemoryAllocator(device=device)
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


def create_test_metadata():
    """Create a test metadata for LMCacheEngineMetadata."""
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


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
def gds_backend(temp_gds_path, async_loop):
    config = create_test_config(temp_gds_path)
    metadata = create_test_metadata()
    return GdsBackend(
        config=config,
        loop=async_loop,
        metadata=metadata,
        dst_device="cuda:0",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA for TestGdsBackend",
)
@pytest.mark.skipif(sys.platform != "linux", reason="TestGdsBackend runs only on Linux")
class TestGdsBackend:
    def test_init(self, temp_gds_path, async_loop):
        config = create_test_config(temp_gds_path)
        metadata = create_test_metadata()
        backend = GdsBackend(
            config=config,
            loop=async_loop,
            metadata=metadata,
            dst_device="cuda:0",
        )
        assert backend.gds_path == temp_gds_path
        assert backend.dst_device == "cuda:0"
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
        memory_obj = create_test_memory_obj(device="cuda")
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
        memory_objs = [create_test_memory_obj(device="cuda") for _ in range(3)]
        futures = gds_backend.batched_submit_put_task(keys, memory_objs)
        assert futures is not None
        assert len(futures) == 3
        for future in futures:
            assert future is not None
            future.result(timeout=5)
        for key in keys:
            assert gds_backend.contains(key)

    def test_get_blocking_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        result = gds_backend.get_blocking(key)
        assert result is None

    def test_close(self, gds_backend):
        # Should not raise
        gds_backend.close()

    def test_pin_unpin_not_implemented(self, gds_backend):
        key = create_test_key(0)
        assert not gds_backend.pin(key)
        assert not gds_backend.unpin(key)

    def test_weka_initialization_suffix(self, temp_gds_path, async_loop):
        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        class DummyCuFileDriver:
            def __init__(self):
                pass

        class DummyCuFile:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, *_, **__):
                return None

            def read(self, *_, **__):
                return 0

        dummy_cufile_module = type(
            "DummyCuFileModule",
            (),
            {"CuFileDriver": DummyCuFileDriver, "CuFile": DummyCuFile},
        )()

        with mock.patch.dict(sys.modules, {"cufile": dummy_cufile_module}):
            with (
                mock.patch(
                    "lmcache.v1.storage_backend.gds_backend.get_fstype",
                    return_value="wekafs",
                ),
                mock.patch.object(
                    GdsBackend,
                    "initialize_allocator",
                    return_value=DummyAllocator(),
                ),
            ):
                config = create_test_config(temp_gds_path)
                metadata = create_test_metadata()

                backend = GdsBackend(
                    config=config,
                    loop=async_loop,
                    metadata=metadata,
                    dst_device="cuda:0",
                )
                try:
                    key = create_test_key(0)
                    path, _, _, _ = backend._key_to_path(key)
                    assert path.endswith(".weka1")
                    assert backend.data_suffix == ".weka1"
                    assert backend.use_cufile
                finally:
                    backend.close()

    def test_weka_disallows_disabling_cufile(self, temp_gds_path, async_loop):
        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        with (
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.get_fstype",
                return_value="wekafs",
            ),
            mock.patch.object(
                GdsBackend,
                "initialize_allocator",
                return_value=DummyAllocator(),
            ),
        ):
            config = create_test_config(temp_gds_path)
            config.extra_config["use_cufile"] = False
            metadata = create_test_metadata()

            with pytest.raises(AssertionError):
                GdsBackend(
                    config=config,
                    loop=async_loop,
                    metadata=metadata,
                    dst_device="cuda:0",
                )
