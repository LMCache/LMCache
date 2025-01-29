import asyncio

import pytest
import torch
from utils import (check_mem_obj_equal, close_asyncio_loop,
                   dumb_cache_engine_key, init_asyncio_loop)

from lmcache.experimental.memory_management import PinMemoryAllocator
from lmcache.experimental.storage_backend.connector import CreateConnector


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
    memory_allocator.ref_count_up(memory_obj)

    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop)
    future.result()

    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key),
                                              async_loop)
    assert future.result()
    assert memory_allocator.get_ref_count(memory_obj) == 1

    future = asyncio.run_coroutine_threadsafe(connector.get(random_key),
                                              async_loop)
    retrieved_memory_obj = future.result()

    check_mem_obj_equal(
        [retrieved_memory_obj],
        [memory_obj],
    )

    close_asyncio_loop(async_loop, async_thread)
