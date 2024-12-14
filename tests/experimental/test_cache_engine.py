import shlex
import subprocess
import time
from copy import deepcopy

import pytest
import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.experimental.gpu_connector import VLLMNestedTupleGPUConnector
from utils import dumb_metadata, generate_kv_cache, generate_tokens


def create_gpu_connector(hidden_dim, num_layers):
    return VLLMNestedTupleGPUConnector(hidden_dim, num_layers)


def test_same_retrieve_store(autorelease):
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
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend="cpu")

    engine = autorelease(
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
    ],
)
def test_retrieve_prefix(fmt, chunk_size, backend, autorelease):
    device = "cuda"
    num_tokens = 8000
    new_num_tokens = 4000
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
    engine = autorelease(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    t1 = time.perf_counter()
    engine.store(tokens, kvcaches=kv_cache)
    t2 = time.perf_counter()
    print(f"store {len(tokens)} takes {t2-t1}")
    """ test retrieve """
    t4 = time.perf_counter()
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache)

    length = torch.sum(ret_mask)
    t5 = time.perf_counter()
    print(f"retrieve {length} takes {t5-t4}")

    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, kv_cache, expected_length, fmt)

    if backend in ["file://local_disk/"]:
        subprocess.run(shlex.split("rm -rf local_disk/"))

    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cuda"])
def test_mixed_retrieve(fmt, chunk_size, backend, autorelease):
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

    engine = autorelease(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    """ test store """
    engine.store(tokens, kvcaches=kv_cache)
    engine.store(new_tokens, kvcaches=new_kv_cache)
    """ test retrieve """
    ret_mask = engine.retrieve(torch.cat([tokens, new_tokens]),
                               kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)

    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrieved_cache, kv_cache, expected_length, fmt)
    """ test another retrieve """
    ret_mask = engine.retrieve(new_tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == new_num_tokens
    check_kv_cache_equal(retrieved_cache, new_kv_cache, length, fmt)
    """ insert the mixed kv cache """
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches(
        [kv_cache, generate_kv_cache(new_num_tokens, fmt, device)], fmt)
    engine.store(final_tokens, kvcaches=final_kv_cache)
    """ should retrieve the mixed version """
    ret_mask = engine.retrieve(final_tokens, kvcaches=retrieved_cache)
    length = torch.sum(ret_mask)
    assert length == num_tokens + new_num_tokens

    check_kv_cache_equal(retrieved_cache, final_kv_cache, length, fmt)
    """destroy local disk path"""
    if backend in ["file://local_disk/"]:
        subprocess.run(shlex.split("rm -rf local_disk/"))

    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
def test_lookup(fmt, autorelease):
    device = "cuda"
    num_tokens = 12000
    new_num_tokens = 2000
    chunk_size = 256
    kv_shape = (32, 2, chunk_size, 8, 128)
    connector = create_gpu_connector(1024, 32)

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)

    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size)
    engine = autorelease(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))

    engine.store(tokens, kvcaches=kv_cache)

    prefix_length = engine.lookup(tokens)
    assert prefix_length == num_tokens, \
        f"Expected {num_tokens} prefix tokens, but got {prefix_length}"

    short_tokens_len = ((num_tokens // 2) // chunk_size) \
        * chunk_size
    short_tokens = tokens[:short_tokens_len]
    prefix_length = engine.lookup(short_tokens)
    assert prefix_length == short_tokens_len, \
        f"Expected {short_tokens_len} prefix tokens, but got {prefix_length}"

    prefix_length = engine.lookup(final_tokens)
    expected_prefix_length = (num_tokens // chunk_size) * chunk_size
    assert prefix_length == expected_prefix_length, \
        f"Expected {expected_prefix_length} prefix tokens,"\
            f" but got {prefix_length}"

    engine.store(final_tokens, kvcaches=final_kv_cache)

    final_prefix_length = engine.lookup(final_tokens)
    assert final_prefix_length == num_tokens + new_num_tokens, \
    f"Expected {num_tokens + new_num_tokens} prefix tokens,"\
        f" but got {final_prefix_length}"

    LMCacheEngineBuilder.destroy("test")


@pytest.mark.parametrize("fmt", ["vllm"])
def test_store_kv_tensors_mask(fmt, autorelease):
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

    engine = autorelease(
        LMCacheEngineBuilder.get_or_create("test", cfg,
                                           dumb_metadata(fmt, kv_shape),
                                           connector))
    ''' Store some tokens with mask '''
    engine.store(tokens, kvcaches=kv_cache)
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


def test_builder(autorelease):
    instance_id = "test"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256,
                                          persist_path="/tmp/a.txt")
    cfg2 = LMCacheEngineConfig.from_legacy(chunk_size=512,
                                           persist_path="/tmp/a.txt")
    connector = None
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    _engine = autorelease(
        LMCacheEngineBuilder.get_or_create(instance_id, cfg, dumb_metadata(),
                                           connector))
    _engine2 = autorelease(LMCacheEngineBuilder.get(instance_id))  # noqa

    with pytest.raises(ValueError):
        LMCacheEngineBuilder.get_or_create(instance_id, cfg2, dumb_metadata(),
                                           connector)
