import asyncio

import pytest
import torch
from utils import (check_mem_obj_equal, close_asyncio_loop,
                   dumb_cache_engine_key, init_asyncio_loop)
import tempfile
from pathlib import Path
from lmcache.experimental.memory_management import PinMemoryAllocator   
from lmcache.experimental.storage_backend.connector import CreateConnector
from lmcache.experimental.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.experimental.storage_backend.connector import parse_remote_url
from lmcache.config import LMCacheEngineConfig


@pytest.mark.parametrize("lmserver_experimental_process", ["cpu"],
                         indirect=True)
@pytest.mark.parametrize(
    "url",
    [
        "lm://localhost:65000",
    ],
)
def test_lm_connector(url, autorelease_experimental,
                      lmserver_experimental_process):
    if url.startswith("lm"):
        url = lmserver_experimental_process.server_url

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    connector = autorelease_experimental(
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

@pytest.mark.parametrize("lmserver_experimental_process", ["cpu"],
                         indirect=True)
def test_fs_connector(lmserver_experimental_process, autorelease_experimental):
    """Test the filesystem connector implementation for storage backend.
    Tests basic operations: exists, put, and get."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure filesystem URL with temporary directory
        url = f"fs://host:0/{temp_dir}/"

        # Create mock config for testing
        mock_config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_device="cpu",
            max_local_cache_size=10,  # max_local_cpu_size in bytes
            remote_url=url,
            remote_serde="naive",
            pipelined_backend=False
        )
        
        # Initialize async event loop and memory management
        async_loop, async_thread = init_asyncio_loop()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)  # 1GB allocation
        connector = autorelease_experimental(
            CreateConnector(url, async_loop, memory_allocator))
        
        # Test 1: Verify key doesn't exist initially
        random_key = dumb_cache_engine_key()
        future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                              async_loop)
        assert not future.result(), "Key should not exist before putting data"

        # Test 2: Create and store test data
        num_tokens = 1000
        mem_obj_shape = [2, 32, num_tokens, 1024]  # [batch, heads, seq_len, hidden_dim]
        dtype = torch.bfloat16
        memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        # Put data into storage
        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop)
        future.result()

        # Test 3: Verify key exists after putting data
        future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                                async_loop)
        assert future.result(), "Key should exist after putting data"
        assert memory_obj.get_ref_count() == 1, "Reference count should be 1"

        # Test 4: Retrieve and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key),
                                                async_loop)
        retrieved_memory_obj = future.result()

        # Verify retrieved data matches original
        check_mem_obj_equal(
            [retrieved_memory_obj],
            [memory_obj],
        )

        # Clean up resources
        close_asyncio_loop(async_loop, async_thread)

