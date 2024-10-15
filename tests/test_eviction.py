import shlex
import subprocess
import time

import pytest
import torch

from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata


def dumb_metadata(fmt="vllm"):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt)


def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = ([num_tokens, num_heads, head_size]
             if fmt == "vllm" else [num_heads, num_tokens, head_size])
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_tokens(num_tokens, device):
    return torch.randint(0, 10000, size=[num_tokens]).to(device)


def concatenate_kv_caches(kv_chunks, fmt):
    dim = 1 if fmt == "huggingface" else 0
    ret = []
    for kv_layer in zip(*kv_chunks):
        klist, vlist = zip(*kv_layer)
        klayer = torch.cat(klist, dim=dim)
        vlayer = torch.cat(vlist, dim=dim)
        ret.append((klayer, vlayer))
    return tuple(ret)


def check_kv_cache_equal(left, right, num_tokens, fmt):
    """
    check if the first num_tokens of left and right kv cache are the same
    """
    dim = 0 if fmt == "vllm" else 1
    for left_kv, right_kv in zip(left, right):
        left_k, left_v = left_kv
        right_k, right_v = right_kv
        right_k = right_k.to(left_k.device)
        right_v = right_v.to(left_v.device)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert left_k.shape[dim] >= num_tokens
        assert left_v.shape[dim] >= num_tokens
        assert right_k.shape[dim] >= num_tokens
        assert right_v.shape[dim] >= num_tokens

        match fmt:
            case "huggingface":
                assert (left_k[:, :num_tokens, :] == right_k[:, :num_tokens, :]
                        ).all()
                assert (left_v[:, :num_tokens, :] == right_v[:, :num_tokens, :]
                        ).all()
            case "vllm":
                assert (left_k[:num_tokens, :, :] == right_k[:num_tokens, :, :]
                        ).all()
                assert (left_v[:num_tokens, :, :] == right_v[:num_tokens, :, :]
                        ).all()


def check_kv_cache_device(kvs, device):
    for kv in kvs:
        k, v = kv
        assert k.device == torch.device(device)
        assert v.device == torch.device(device)


# @pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
# def test_func_get_locations(fmt, autorelease):
#     device = "cuda"
#     num_tokens = 12000
#     new_num_tokens = 200
#     chunk_size = 256
#     persist_path = "/tmp/test-engine.pth"

#     tokens = generate_tokens(num_tokens, device)
#     kv_cache = generate_kv_cache(num_tokens, fmt, device)
#     new_tokens = generate_tokens(new_num_tokens, device)
#     new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
#     final_tokens = torch.cat([tokens, new_tokens])
#     final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
#     """ initialize the engine """
#     cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
#                                           persist_path=persist_path)
#     engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     """ store and persist """
#     engine1.store(tokens, kv_cache)
#     engine2.store(tokens, kv_cache)

#     """ check locations """
#     location_list = engine1.get_locations(tokens)
#     assert(len(location_list) == (num_tokens - 1) // chunk_size  + 1)
#     for location in location_list:
#         assert (location == ['local cuda'])

# @pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
# def test_func_get_locations_cpu(fmt, autorelease):
#     device = "cuda"
#     num_tokens = 12000
#     new_num_tokens = 200
#     chunk_size = 256
#     persist_path = "/tmp/test-engine.pth"

#     tokens = generate_tokens(num_tokens, device)
#     kv_cache = generate_kv_cache(num_tokens, fmt, device)
#     new_tokens = generate_tokens(new_num_tokens, device)
#     new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
#     final_tokens = torch.cat([tokens, new_tokens])
#     final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
#     """ initialize the engine """
#     cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
#                                           persist_path=persist_path,
#                                           backend='cpu')
#     engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     """ store and persist """
#     engine1.store(tokens, kv_cache)
#     engine2.store(tokens, kv_cache)

#     """ check locations """
#     location_list = engine1.get_locations(tokens)
#     assert(len(location_list) == (num_tokens - 1) // chunk_size  + 1)
#     for location in location_list:
#         assert (location == ['local cpu'])

# @pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
# def test_func_get_locations_cpu(fmt, autorelease):
#     device = "cuda"
#     num_tokens = 12000
#     new_num_tokens = 200
#     chunk_size = 256
#     persist_path = "/tmp/test-engine.pth"

#     tokens = generate_tokens(num_tokens, device)
#     kv_cache = generate_kv_cache(num_tokens, fmt, device)
#     new_tokens = generate_tokens(new_num_tokens, device)
#     new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
#     final_tokens = torch.cat([tokens, new_tokens])
#     final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
#     """ initialize the engine """
#     cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
#                                           persist_path=persist_path,
#                                           backend='cpu')
#     engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     """ store and persist """
#     engine1.store(tokens, kv_cache)
#     engine2.store(tokens, kv_cache)

#     """ check locations """
#     location_list = engine1.get_locations(tokens)
#     assert(len(location_list) == (num_tokens - 1) // chunk_size  + 1)
#     for location in location_list:
#         assert (location == ['local cpu'])

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_func_get_locations_remove_cpu(fmt, autorelease):
    device = "cuda"
    num_tokens = 12000
    new_num_tokens = 200
    chunk_size = 256
    persist_path = "/tmp/test-engine.pth"

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
    """ initialize the engine """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
                                          persist_path=persist_path,
                                          backend='cpu')
    engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
    engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
    """ store and persist """
    engine1.store(tokens, kv_cache)
    engine2.store(tokens, kv_cache)

    """ check locations """
    location_list = engine1.get_locations(tokens)
    assert(len(location_list) == (num_tokens - 1) // chunk_size  + 1)
    for location in location_list:
        assert (location == ['local cpu'])
    
    engine1.remove(tokens[:chunk_size], ['local cpu'])
    assert (engine1.get_locations(tokens)[0] == None )

    saved = engine1.remove(tokens[:chunk_size], ['local cpu'])
    assert(saved[1] == [False])

    saved = engine1.remove(tokens[:chunk_size+10], ['local cpu'])
    assert(saved[0] == False)


# @pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
# def test_func_retrieve_not_here(fmt, autorelease):
#     device = "cuda"
#     num_tokens = 12000
#     new_num_tokens = 200
#     chunk_size = 256
#     persist_path = "/tmp/test-engine.pth"

#     tokens = generate_tokens(num_tokens, device)
#     kv_cache = generate_kv_cache(num_tokens, fmt, device)
#     new_tokens = generate_tokens(new_num_tokens, device)
#     new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
#     final_tokens = torch.cat([tokens, new_tokens])
#     final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
#     """ initialize the engine """
#     cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
#                                           persist_path=persist_path)
#     engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     """ store and persist """
#     engine1.store(tokens, kv_cache)
#     engine2.store(tokens, kv_cache)
#     """ store final """
#     t1 = time.perf_counter()
#     engine1.store(final_tokens, final_kv_cache, skip_existing=True)
#     t2 = time.perf_counter()
#     engine2.store(final_tokens, final_kv_cache, skip_existing=False)
#     t3 = time.perf_counter()

# @pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
# def test_func_retrieve_wrong_block_fmt(fmt, autorelease):
#     device = "cuda"
#     num_tokens = 12000
#     new_num_tokens = 200
#     chunk_size = 256
#     persist_path = "/tmp/test-engine.pth"

#     tokens = generate_tokens(num_tokens, device)
#     kv_cache = generate_kv_cache(num_tokens, fmt, device)
#     new_tokens = generate_tokens(new_num_tokens, device)
#     new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
#     final_tokens = torch.cat([tokens, new_tokens])
#     final_kv_cache = concatenate_kv_caches([kv_cache, new_kv_cache], fmt)
#     """ initialize the engine """
#     cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size,
#                                           persist_path=persist_path)
#     engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
#     """ store and persist """
#     engine1.store(tokens, kv_cache)
#     engine2.store(tokens, kv_cache)
#     """ store final """
#     t1 = time.perf_counter()
#     engine1.store(final_tokens, final_kv_cache, skip_existing=True)
#     t2 = time.perf_counter()
#     engine2.store(final_tokens, final_kv_cache, skip_existing=False)
#     t3 = time.perf_counter()