# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
import random
import shlex
import subprocess
import time

# Third Party
from utils import (
    check_paged_kv_cache_equal,
    create_gpu_connector,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
)
import pytest
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig


def test_paged_same_retrieve_store(autorelease_v1):
    device = "cuda"
    fmt = "vllm"
    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)

    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    original_retrieved_cache = deepcopy(retrieved_cache)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    # Check the kv cache and the retrieval buffer are not the same
    check_paged_kv_cache_equal(retrieved_cache, original_retrieved_cache, slot_mapping)
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(retrieved_cache, kv_cache, slot_mapping)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, remote_url=None)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test retrieve empty """
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    length = torch.sum(ret_mask)
    assert length == 0
    check_paged_kv_cache_equal(retrieved_cache, original_retrieved_cache, slot_mapping)
    """ test store """
    engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)

    """ Store is async. Need to wait for the store to finish """
    timeout = 1.5
    start_time = time.time()
    while engine.lookup(tokens) < num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    length = torch.sum(ret_mask)
    assert length == num_tokens
    check_paged_kv_cache_equal(retrieved_cache, kv_cache, slot_mapping)


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cpu", "local_disk", "remote", "remote_cachegen"])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_retrieve_prefix(
    fmt, chunk_size, backend, lmserver_v1_process, autorelease_v1
):
    url = None
    remote_serde = None
    check_equality = True
    if "remote" in backend:
        url = lmserver_v1_process.server_url
        if backend == "remote_cachegen":
            backend = "remote"
            remote_serde = "cachegen"
            check_equality = False
        else:
            remote_serde = "naive"
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size,
        backend=backend,
        remote_url=url,
        remote_serde=remote_serde,
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
    """ Compute expected length """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
        search_range = "LocalCPUBackend"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    elif backend == "remote":
        timeout = 30
        search_range = "RemoteBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == expected_length

    if check_equality:
        check_paged_kv_cache_equal(
            kv_cache,
            retrieved_cache,
            torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
        )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize(
    "backend",
    ["cpu", "local_disk", "remote"],
)
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_store_offset(
    fmt, chunk_size, backend, lmserver_v1_process, autorelease_v1
):
    url = None
    if backend == "remote":
        url = lmserver_v1_process.server_url
    device = "cuda"
    num_tokens = 2000
    num_suffix_tokens = 500
    num_total_tokens = 3000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_total_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_total_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, backend=backend, remote_url=url
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    engine.store(
        tokens[:num_tokens],
        kvcaches=kv_cache,
        slot_mapping=slot_mapping[:num_tokens],
    )

    offset_chunk_cnt = num_tokens // chunk_size
    offset_length = offset_chunk_cnt * chunk_size
    mask = torch.ones(num_tokens + num_suffix_tokens, device=device)
    mask[:offset_length] = 0
    engine.store(
        tokens[: num_tokens + num_suffix_tokens],
        kvcaches=kv_cache,
        mask=mask,
        slot_mapping=slot_mapping[: num_tokens + num_suffix_tokens],
    )
    """ Compute expected length """
    expected_chunk_cnt = (num_tokens + num_suffix_tokens) // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    """ Store is async. Need to wait for the store to finish """
    if backend == "cpu":
        timeout = 1
    elif backend == "local_disk":
        timeout = 30
    start_time = time.time()
    while engine.lookup(tokens[: num_tokens + num_suffix_tokens]) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        kv_cache,
        retrieved_cache,
        slot_mapping[:expected_length],
    )

    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128])  # , 256])
@pytest.mark.parametrize(
    "backend",
    [
        # "cpu",
        "local_disk"
    ],
)
def test_paged_mixed_retrieve(fmt, chunk_size, backend, autorelease_v1):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend=backend)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    engine.store(new_tokens, kvcaches=kv_cache, slot_mapping=new_slot_mapping)
    """ Store is async. Need to wait for the store to finish """
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    if backend == "cpu":
        timeout = 1
        search_range = "LocalCPUBackend"
    elif backend == "local_disk":
        timeout = 30
        search_range = "LocalDiskBackend"
    start_time = time.time()
    while engine.lookup(tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test retrieve """
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    length = torch.sum(ret_mask)
    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
    )

    """Wait for store to finish"""
    expected_length = new_num_tokens
    start_time = time.time()
    while engine.lookup(new_tokens, search_range) < expected_length:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ test another retrieve """
    ret_mask = engine.retrieve(
        new_tokens, kvcaches=retrieved_cache, slot_mapping=new_slot_mapping
    )
    length = torch.sum(ret_mask)
    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache, kv_cache, new_slot_mapping[:expected_length]
    )

    """ insert the mixed kv cache """
    final_tokens = torch.cat([tokens, new_tokens])
    engine.store(
        final_tokens,
        kvcaches=kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )

    """Wait until store finishes"""
    expected_length = num_tokens + new_num_tokens
    start_time = time.time()
    while (
        engine.lookup(torch.cat([tokens, new_tokens]), search_range) < expected_length
    ):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)
    """ should retrieve the mixed version """
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    length = torch.sum(ret_mask)
    assert length == expected_length

    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    """destroy local disk path"""
    if backend in ["local_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm"])
def test_paged_store_kv_tensors_mask(fmt, autorelease_v1):
    device = "cuda"
    num_tokens = 1000
    new_num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    new_tokens = generate_tokens(new_num_tokens, device)
    final_tokens = torch.cat([tokens, new_tokens])

    slot_mapping_full = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping_full[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping_full[-new_num_tokens:], device=device)

    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ Store some tokens with mask """
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    """Wait until store finishes"""
    timeout = 1
    start_time = time.time()
    while engine.lookup(tokens) < num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(tokens)
    assert prefix_length == num_tokens, (
        f"Expected {num_tokens} prefix tokens, but got {prefix_length}"
    )
    """ Store more tokens """
    prefix_length = engine.lookup(final_tokens)
    kv_tensor_mask = torch.ones_like(final_tokens, dtype=torch.bool)
    kv_tensor_mask[:prefix_length] = False

    engine.store(
        final_tokens,
        mask=kv_tensor_mask,
        kvcaches=kv_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    """Wait until store finishes"""
    start_time = time.time()
    while engine.lookup(final_tokens) < num_tokens + new_num_tokens:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
        time.sleep(0.01)

    prefix_length = engine.lookup(final_tokens)
    assert prefix_length == num_tokens + new_num_tokens, (
        f"Expected {num_tokens + new_num_tokens} prefix tokens, but got {prefix_length}"
    )
    """ retrieve the whole cache """
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    length = torch.sum(ret_mask)
    expected_length = num_tokens + new_num_tokens
    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
    )

    """ retrieve cache with some mask:
    """
    num_falses = chunk_size * 3
    mask = torch.ones_like(final_tokens, dtype=torch.bool)
    mask[:num_falses] = False
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    ret_mask = engine.retrieve(
        final_tokens,
        mask=mask,
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )
    length = torch.sum(ret_mask)
    full_length = num_tokens + new_num_tokens
    expected_length = full_length - num_falses
    assert length == expected_length

    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            retrieved_cache,
            kv_cache,
            torch.cat([slot_mapping, new_slot_mapping])[:full_length],
        )
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[num_falses:full_length],
    )

    mask[: num_falses + 5] = False
    with pytest.raises(ValueError):
        engine.retrieve(
            final_tokens,
            mask=mask,
            kvcaches=retrieved_cache,
            slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
        )


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize(
    "backend",
    [
        "local_cpu_disk_remote",
    ],
)
@pytest.mark.parametrize(
    "retrieve_from",
    [
        "local_cpu",
        "local_disk",
        "remote",
    ],
)
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_hierarchy_retrieve(
    fmt, chunk_size, backend, retrieve_from, lmserver_v1_process, autorelease_v1
):
    url = None
    if backend == "local_cpu_disk_remote":
        url = lmserver_v1_process.server_url
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    slot_mapping = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, backend=backend, remote_url=url
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
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
    if retrieve_from in ["local_disk", "remote"]:
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens, ["LocalDiskBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """ Wait until remote save is finished """
    if retrieve_from == "remote":
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        # FIXME: change this `clear`
        engine.storage_manager.storage_backends["LocalDiskBackend"].dict.clear()
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens, ["RemoteBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
    )

    """ Wait until disk save is finished before deleting the directory"""
    if backend in ["local_cpu_disk"]:
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


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
def test_paged_prefetch_retrieve(backend, prefetch_from, autorelease_v1):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16

    chunk_size = 256
    fmt = "vllm"
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )
    new_tokens = generate_tokens(new_num_tokens, device)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype=dtype
    )

    slot_mapping = random.sample(
        range(0, num_blocks * block_size), num_tokens + new_num_tokens
    )
    slot_mapping = torch.tensor(slot_mapping[:num_tokens], device=device)

    new_slot_mapping = torch.tensor(slot_mapping[-new_num_tokens:], device=device)

    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend=backend)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2 - t1}")
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
        engine.storage_manager.clear(locations=["LocalCPUBackend"])
        timeout = 30
        start_time = time.time()
        while engine.lookup(tokens) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.1)
    """ Wait until disk load (prefetch) finishes and delete disk cache"""
    engine.prefetch(torch.cat([tokens, new_tokens]))

    if prefetch_from == "local_disk":
        timeout = 60
        start_time = time.time()
        while (
            engine.lookup(torch.cat([tokens, new_tokens]), ["LocalCPUBackend"])
            < expected_length
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
        engine.storage_manager.storage_backends["LocalDiskBackend"].dict.clear()
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(
        torch.cat([tokens, new_tokens]),
        kvcaches=retrieved_cache,
        slot_mapping=torch.cat([slot_mapping, new_slot_mapping]),
    )

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5 - t4}")

    assert length == expected_length
    check_paged_kv_cache_equal(
        retrieved_cache,
        kv_cache,
        torch.cat([slot_mapping, new_slot_mapping])[:expected_length],
    )

    if backend in ["local_cpu_disk"]:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        "local_disk",
        "remote",
        "local_disk_remote",
        "local_cpu_disk_remote",
    ],
)
@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_mem_leak(fmt, chunk_size, backend, lmserver_v1_process, autorelease_v1):
    url = None
    if "remote" in backend:
        url = lmserver_v1_process.server_url

    device = "cuda"
    num_tokens = 2000
    kv_shape = (32, 2, chunk_size, 8, 128)
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, backend=backend, remote_url=url
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)

    expected_length = 2000
    timeout = 30
    """Wait until cpu store finishes"""
    if "cpu" in backend:
        start_time = time.time()
        while engine.lookup(tokens, ["LocalCPUBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    """Wait until disk store finishes"""
    if "disk" in backend:
        start_time = time.time()
        while engine.lookup(tokens, ["LocalDiskBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)

    if "remote" in backend:
        start_time = time.time()
        while engine.lookup(tokens, ["RemoteBackend"]) < expected_length:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds.")
            time.sleep(0.01)
    tensor_memory_allocator = (
        engine.storage_manager.allocator_backend.memory_allocator.pin_allocator
    )
    if "cpu" not in backend:
        assert tensor_memory_allocator.total_allocated_size == 0
    else:
        assert tensor_memory_allocator.total_allocated_size > 0

    if "disk" in backend:
        subprocess.run(shlex.split("rm -rf local/disk_test/local_disk/"))


def test_builder(autorelease_v1):
    instance_id = "test"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256)
    cfg2 = LMCacheEngineConfig.from_legacy(chunk_size=512)
    connector = None
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    _engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg,
            dumb_metadata(),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )
    _engine2 = autorelease_v1(LMCacheEngineBuilder.get(instance_id))  # noqa

    with pytest.raises(ValueError):
        LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg2,
            dumb_metadata(),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
