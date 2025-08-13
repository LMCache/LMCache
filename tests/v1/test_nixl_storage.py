# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PagedTensorMemoryAllocator
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.nixl_storage_backend import NixlStorageBackend


def run(config: LMCacheEngineConfig, shape, dtype):
    BACKEND_NAME = "NixlStorageBackend"
    TEST_KEY = CacheEngineKey(
        fmt="MemoryFormat.KV_2LTD",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash=int(
            "e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157", base=16
        ),
    )
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        metadata = LMCacheEngineMetadata(
            "TinyLlama-1.1B-Chat-v1.0",
            0,
            0,
            "MemoryFormat.KV_2LTD",
            dtype,
            shape,
            False,
        )

        backends = CreateStorageBackends(
            config,
            metadata,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config, metadata),
        )
        assert len(backends) == 2  # NixlStorageBackend + LocalCPUBackend
        assert BACKEND_NAME in backends

        nixl_backend = backends[BACKEND_NAME]
        assert isinstance(nixl_backend, NixlStorageBackend)
        assert isinstance(nixl_backend.memory_allocator, PagedTensorMemoryAllocator)
        assert nixl_backend is not None
        assert nixl_backend.memory_allocator is not None

        assert not nixl_backend.contains(TEST_KEY, False)
        assert not nixl_backend.exists_in_put_tasks(TEST_KEY)

        memory_obj = nixl_backend.memory_allocator.allocate(shape=shape, dtype=dtype)
        assert memory_obj is not None
        assert memory_obj.tensor is not None

        # small tensor changes for data validation
        memory_obj.tensor[100, 200] = 1e-3
        memory_obj.tensor[200, 100] = 1e-4

        nixl_backend.batched_submit_put_task([TEST_KEY], [memory_obj])

        returned_memory_obj = nixl_backend.get_blocking(TEST_KEY)
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()
        assert returned_memory_obj.metadata.address != memory_obj.metadata.address
        assert torch.equal(returned_memory_obj.tensor, memory_obj.tensor)
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()


@pytest.mark.no_shared_allocator
def test_nixl_gds_mt_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda"
    config.extra_config["nixl_backend"] = "GDS_MT"

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_mt_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS_MT"

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_cuda_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cuda"
    config.extra_config["nixl_backend"] = "GDS"

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_gds_cpu_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS"

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = [2048, 2048]

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"

    run(config, shape, dtype)
