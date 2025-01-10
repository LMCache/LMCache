import random
import shlex
import subprocess
import time
from copy import deepcopy

import pytest
import torch
from utils import (check_kv_cache_equal, check_paged_kv_cache_equal,
                   concatenate_kv_caches, create_gpu_connector, dumb_metadata,
                   generate_kv_cache, generate_kv_cache_paged, generate_tokens)

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.experimental.config import LMCacheEngineConfig


def test_same_retrieve_store(autorelease_experimental):
    device = "cuda"
    fmt = "vllm"
    num_tokens = 2000
    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)

    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    retrieved_cache = generate_kv_cache(num_tokens, fmt, device)
    original_retrieved_cache = deepcopy(retrieved_cache)

    # Check the kv cache and the retrieval buffer are not the same
    check_kv_cache_equal(retrieved_cache, original_retrieved_cache, num_tokens,
                         fmt)
    with pytest.raises(AssertionError):
        check_kv_cache_equal(retrieved_cache, kv_cache, num_tokens, fmt)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test retrieve empty """
    ret_mask = engine.retrieve(tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == 0
    check_kv_cache_equal(retrieved_cache, original_retrieved_cache, num_tokens,
                         fmt)
    """ test store """
    engine.store(tokens, kvcaches=kv_cache)
    """ Store is async. Need to wait for the store to finish """
    timeout = 1.5
    start_time = time.time()
    while engine.lookup(tokens) < num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    ret_mask = engine.retrieve(tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)

    assert length == num_tokens
    check_kv_cache_equal(retrieved_cache, kv_cache, num_tokens, fmt)

    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        "local_disk",
    ],
)
def test_paged_retrieve_prefix(fmt, chunk_size, backend,
                               autorelease_experimental):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32, paged=True)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged(num_blocks, device, block_size, dtype)
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = kv_cache = generate_kv_cache_paged(
        num_blocks, device, block_size, dtype)
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    new_slot_mapping = random.sample(range(0, num_blocks * block_size),
                                     new_num_tokens)
    new_slot_mapping = torch.tensor(new_slot_mapping, device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          backend=backend)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2-t1}")
    """ Compute expected length """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
        search_range = "Hot"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache,
                               slot_mapping=torch.cat(
                                   [slot_mapping, new_slot_mapping]))

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5-t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        kv_cache,
        retrieved_cache,
        num_tokens,
        slot_mapping,
    )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf /local/disk_test/local_disk/"))
    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize(
    "backend",
    ["cpu", "local_disk"],
)
def test_paged_store_offset(fmt, chunk_size, backend,
                            autorelease_experimental):
    device = "cuda"
    num_tokens = 2000
    num_suffix_tokens = 500
    num_total_tokens = 3000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32, paged=True)

    tokens = generate_tokens(num_total_tokens, device)
    kv_cache = generate_kv_cache_paged(num_blocks, device, block_size, dtype)
    retrieved_cache = kv_cache = generate_kv_cache_paged(
        num_blocks, device, block_size, dtype)
    slot_mapping = random.sample(range(0, num_blocks * block_size),
                                 num_total_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          backend=backend)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    engine.store(tokens[:num_tokens],
                 kvcaches=kv_cache,
                 slot_mapping=slot_mapping[:num_tokens])

    offset_chunk_cnt = num_tokens // chunk_size
    offset_length = offset_chunk_cnt * chunk_size
    mask = torch.ones(num_tokens + num_suffix_tokens, device=device)
    mask[:offset_length] = 0
    engine.store(tokens[:num_tokens+num_suffix_tokens],
                 kvcaches=kv_cache,
                 mask=mask,
                 slot_mapping=slot_mapping[offset_length: \
                     num_tokens+num_suffix_tokens],
                 offset=offset_length)
    """ Compute expected length """
    expected_chunk_cnt = (num_tokens + num_suffix_tokens) // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
    elif backend == "local_disk":
        timeout = 30
    start_time = time.time()
    while engine.lookup(tokens[:num_tokens+num_suffix_tokens])\
        < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(tokens,
                               kvcaches=retrieved_cache,
                               slot_mapping=slot_mapping)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5-t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        kv_cache,
        retrieved_cache,
        num_tokens,
        slot_mapping,
    )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf /local/disk_test/local_disk/"))
    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128])  #, 256])
@pytest.mark.parametrize(
    "backend",
    [
        #"cpu",
        "local_disk"
    ])
def test_mixed_retrieve(fmt, chunk_size, backend, autorelease_experimental):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000

    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    retrieved_cache = generate_kv_cache(num_tokens + new_num_tokens, fmt,
                                        device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          backend=backend)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    engine.store(tokens, kvcaches=kv_cache)
    engine.store(new_tokens, kvcaches=new_kv_cache)
    """ Store is async. Need to wait for the store to finish """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    if backend == "cpu":
        timeout = 1
        search_range = "Hot"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, kv_cache, expected_length, fmt)
    """Wait for store to finish"""
    expected_length = new_num_tokens
    start_time = time.time()
    while engine.lookup(new_tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test another retrieve """
    ret_mask = engine.retrieve(new_tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, new_kv_cache, expected_length, fmt)
    """ insert the mixed kv cache """
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches(
        [kv_cache, generate_kv_cache(new_num_tokens, fmt, device)], fmt)
    engine.store(final_tokens, kvcaches=final_kv_cache)
    """Wait until store finishes"""
    expected_length = num_tokens + new_num_tokens
    start_time = time.time()
    while engine.lookup(torch.cat([tokens, new_tokens]),
                        search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ should retrieve the mixed version """
    ret_mask = engine.retrieve(final_tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == expected_length

    check_kv_cache_equal(retrieved_cache, final_kv_cache, expected_length, fmt)
    """destroy local disk path"""
    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf /local/disk_test/local_disk/"))

    #engine.close()
    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
def test_store_kv_tensors_mask(fmt, autorelease_experimental):
    device = "cuda"
    num_tokens = 1000
    new_num_tokens = 2000
    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    final_tokens = torch.cat([tokens, new_tokens])

    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    ''' Store some tokens with mask '''
    engine.store(tokens, kvcaches=kv_cache)
    """Wait until store finishes"""
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(tokens)
    assert prefix_length == num_tokens, \
        f"Expected {num_tokens} prefix tokens, but got {prefix_length}"
    ''' Store more tokens '''
    prefix_length = engine.lookup(final_tokens)
    kv_tensor_mask = torch.ones_like(final_tokens, dtype=torch.bool)
    kv_tensor_mask[:prefix_length] = False

    more_cache_tokens = num_tokens + new_num_tokens - prefix_length
    more_kv_cache = generate_kv_cache(more_cache_tokens, fmt, device)
    concated_kv_cache = concatenate_kv_caches([kv_cache, more_kv_cache], fmt)
    engine.store(final_tokens, mask=kv_tensor_mask, kvcaches=concated_kv_cache)
    """Wait until store finishes"""
    start_time = time.time()
    while engine.lookup(final_tokens) < num_tokens + new_num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(final_tokens)
    assert prefix_length == num_tokens + new_num_tokens, \
        f"Expected {num_tokens + new_num_tokens} prefix tokens,"\
            f" but got {prefix_length}"
    ''' retrieve the whole cache '''
    retrieved_cache = generate_kv_cache(num_tokens + new_num_tokens, fmt,
                                        device)
    ret_mask = engine.retrieve(final_tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == num_tokens + new_num_tokens
    check_kv_cache_equal(retrieved_cache,
                         concatenate_kv_caches([kv_cache, more_kv_cache], fmt),
                         num_tokens, fmt)
    ''' retrieve cache with some mask:
    '''
    num_falses = chunk_size * 3
    mask = torch.ones_like(final_tokens, dtype=torch.bool)
    mask[:num_falses] = False
    retrieved_cache = generate_kv_cache(num_tokens + new_num_tokens, fmt,
                                        device)
    ret_mask = engine.retrieve(final_tokens,
                               mask=mask,
                               kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == num_tokens + new_num_tokens - num_falses
    final_kv_cache = concatenate_kv_caches([kv_cache, more_kv_cache], fmt)

    with pytest.raises(AssertionError):
        check_kv_cache_equal(retrieved_cache, final_kv_cache, num_tokens, fmt)
    check_kv_cache_equal(retrieved_cache,
                         final_kv_cache,
                         num_tokens - num_falses,
                         fmt,
                         offset=num_falses)

    mask[:num_falses + 5] = False
    with pytest.raises(ValueError):
        engine.retrieve(final_tokens, mask=mask, kvcaches=retrieved_cache)

    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize(
    "backend",
    [
        "local_cpu_disk",
    ],
)
@pytest.mark.parametrize(
    "retrieve_from",
    [
        "local_cpu",
        "local_disk",
    ],
)
def test_hierarchy_retrieve(fmt, chunk_size, backend, retrieve_from,
                            autorelease_experimental):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache(new_num_tokens + num_tokens, fmt,
                                        device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          backend=backend)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2-t1}")
    """ Compute expected length """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Store is async. Need to wait for the store to finish """
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ Wait until disk save is finished """
    if retrieve_from == "local_disk":
        engine.storage_manager.hot_cache.clear()
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5-t4}")

    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, kv_cache, expected_length, fmt)
    """ Wait until disk save is finished before deleting the directory"""
    if backend in ["local_cpu_disk"]:
        engine.storage_manager.hot_cache.clear()
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf /local/disk_test/local_disk/"))


@pytest.mark.parametrize(
    "backend",
    [
        "local_cpu_disk",
    ],
)
@pytest.mark.parametrize(
    "prefetch_from",
    [
        "local_disk",
    ],
)
def test_prefetch_retrieve(backend, prefetch_from, autorelease_experimental):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    chunk_size = 256
    fmt = "vllm"
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache(new_num_tokens + num_tokens, fmt,
                                        device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          backend=backend)

    engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2-t1}")
    """ Compute expected length """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Wait for cpu store to finish """
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ Delete cpu cache and wait until disk save finishes."""
    if prefetch_from == "local_disk":
        engine.storage_manager.hot_cache.clear()
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Operation timed out after {timeout} seconds.")
            time.sleep(0.1)
    """ Wait until disk load (prefetch) finishes and delete disk cache"""
    engine.prefetch(torch.cat([tokens, new_tokens]))

    if prefetch_from == "local_disk":
        timeout = 60
        start_time = time.time()
        while engine.lookup(torch.cat([tokens, new_tokens]),
                            ["Hot"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
        engine.storage_manager.storage_backends["LocalDiskBackend"].dict.clear(
        )
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5-t4}")

    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, kv_cache, expected_length, fmt)

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf /local/disk_test/local_disk/"))
    LMCacheEngineBuilder.destroy("test")


def test_builder(autorelease_experimental):
    instance_id = "test"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256)
    cfg2 = LMCacheEngineConfig.from_legacy(chunk_size=512)
    connector = None
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    _engine = autorelease_experimental(
        LMCacheEngineBuilder.get_or_create(instance_id, cfg, dumb_metadata(),
                                           connector))
    _engine2 = autorelease_experimental(
        LMCacheEngineBuilder.get(instance_id))  # noqa

    with pytest.raises(ValueError):
        LMCacheEngineBuilder.get_or_create(instance_id, cfg2, dumb_metadata(),
                                           connector)
