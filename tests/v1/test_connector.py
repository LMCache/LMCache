import asyncio
import tempfile
from pathlib import Path

import pytest
import torch
from utils import (check_mem_obj_equal, close_asyncio_loop,
                   dumb_cache_engine_key, init_asyncio_loop)

from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache.v1.storage_backend.connector import CreateConnector


@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "url",
    [
        "lm://localhost:65000",
    ],
)
def test_lm_connector(url, autorelease_v1, lmserver_v1_process):
    if url.startswith("lm"):
        url = lmserver_v1_process.server_url

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    connector = autorelease_v1(
        CreateConnector(url, async_loop, memory_allocator))

    random_key = dumb_cache_engine_key()
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                              async_loop)
    assert not future.result()

    num_tokens = 1000
    mem_obj_shape = [2, 32, num_tokens, 1024]
    dtype = torch.bfloat16
    memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()

    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop)
    future.result()

    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                              async_loop)
    assert future.result()
    assert memory_obj.get_ref_count() == 1

    future = asyncio.run_coroutine_threadsafe(connector.get(random_key),
                                              async_loop)
    retrieved_memory_obj = future.result()

    check_mem_obj_equal(
        [retrieved_memory_obj],
        [memory_obj],
    )

    close_asyncio_loop(async_loop, async_thread)


@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_fs_connector(lmserver_v1_process, autorelease_v1):
    """Test filesystem connector: exists, put, get, list, and file store."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        url = f"fs://host:0/{temp_dir}/"
        async_loop, async_thread = init_asyncio_loop()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        connector = autorelease_v1(
            CreateConnector(url, async_loop, memory_allocator))
        random_key = dumb_cache_engine_key()

        # Test 1: Verify key doesn't exist initially
        future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                                  async_loop)
        assert not future.result()

        # Test 2: Create and store test data
        dtype = torch.bfloat16
        memory_obj = memory_allocator.allocate([2, 32, 1000, 1024], dtype)
        memory_obj.ref_count_up()
        # Fill with deterministic test data
        torch.manual_seed(42)
        test_tensor = torch.randint(0,
                                    100,
                                    memory_obj.raw_data.shape,
                                    dtype=torch.int64)
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop)
        future.result()

        # Test 3: Verify key exists after putting data
        future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                                  async_loop)
        assert future.result()
        assert memory_obj.get_ref_count() == 1

        # Test 4: Retrieve and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key),
                                                  async_loop)
        check_mem_obj_equal([future.result()], [memory_obj])

        # Test 5: List the keys
        future = asyncio.run_coroutine_threadsafe(connector.list(), async_loop)
        assert future.result() == [random_key.to_string()]

        # Test 6: Verify file existence and format
        files = list(Path(temp_dir).glob("*.data"))
        assert len(files) == 1
        assert files[0].name == f"{random_key.to_string()}.data"

        close_asyncio_loop(async_loop, async_thread)
