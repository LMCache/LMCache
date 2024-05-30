import pytest
import time
import os
import torch
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineConfig, LMCacheEngineBuilder

def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size] if fmt == "vllm" else [num_heads, num_tokens, head_size]
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(32):
        k = torch.rand(shape, dtype = dtype, device = device)
        v = torch.rand(shape, dtype = dtype, device = device)
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
                assert (left_k[:, :num_tokens, :] == right_k[:, :num_tokens, :]).all()
                assert (left_v[:, :num_tokens, :] == right_v[:, :num_tokens, :]).all()
            case "vllm":
                assert (left_k[:num_tokens, :, :] == right_k[:num_tokens, :, :]).all()
                assert (left_v[:num_tokens, :, :] == right_v[:num_tokens, :, :]).all()

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_same_retrive_store(fmt):
    device = "cuda"
    num_tokens = 8000

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_defaults(chunk_size = 256)
    engine = LMCacheEngine(cfg)

    ''' test store '''
    engine.store(tokens, kv_cache, fmt)

    ''' test retrive '''
    retrived_cache, length = engine.retrive(tokens, fmt, device)

    assert length == num_tokens
    check_kv_cache_equal(retrived_cache, kv_cache, num_tokens, fmt)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [128, 256])
def test_retrive_prefix(fmt, chunk_size):
    device = "cuda"
    num_tokens = 8000
    new_num_tokens = 2000

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    engine = LMCacheEngine(cfg)

    ''' test store '''
    engine.store(tokens, kv_cache, fmt)

    ''' test retrive '''
    retrived_cache, length = engine.retrive(torch.cat([tokens, new_tokens]), fmt, device)

    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrived_cache, kv_cache, expected_length, fmt)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [128, 256])
def test_mixed_retrive(fmt, chunk_size):
    device = "cuda"
    num_tokens = 8000
    new_num_tokens = 2000

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    engine = LMCacheEngine(cfg)

    ''' test store '''
    engine.store(tokens, kv_cache, fmt)
    engine.store(new_tokens, new_kv_cache, fmt)

    ''' test retrive '''
    retrived_cache, length = engine.retrive(torch.cat([tokens, new_tokens]), fmt, device)

    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrived_cache, kv_cache, expected_length, fmt)
    
    ''' test another retrive '''
    retrived_cache, length = engine.retrive(new_tokens, fmt, device)

    assert length == new_num_tokens
    check_kv_cache_equal(retrived_cache, new_kv_cache, length, fmt)

    ''' insert the mixed kv cache '''
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches([kv_cache, generate_kv_cache(new_num_tokens, fmt, device)], fmt)
    engine.store(final_tokens, final_kv_cache, fmt)

    ''' should retrive the mixed version '''
    retrived_cache, length = engine.retrive(final_tokens, fmt, device)
    assert length == num_tokens + new_num_tokens
    check_kv_cache_equal(retrived_cache, final_kv_cache, length, fmt)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_persist(fmt):
    device = "cuda"
    num_tokens = 1000
    chunk_size = 256
    persist_path = "/tmp/test-engine.pth"

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size, persist_path = persist_path)
    engine = LMCacheEngine(cfg)

    ''' store and persist '''
    engine.store(tokens, kv_cache, fmt)
    engine.persist()

    ''' test load and retrive '''
    engine2 = LMCacheEngine(cfg)

    retrived_cache, length = engine2.retrive(tokens, fmt, device)

    assert length == num_tokens
    check_kv_cache_equal(retrived_cache, kv_cache, num_tokens, fmt)
    
    os.remove(persist_path)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_skipping(fmt):
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
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size, persist_path = persist_path)
    engine1 = LMCacheEngine(cfg)
    engine2 = LMCacheEngine(cfg)

    ''' store and persist '''
    engine1.store(tokens, kv_cache, fmt)
    engine2.store(tokens, kv_cache, fmt)

    ''' store final '''
    t1 = time.perf_counter()
    engine1.store(final_tokens, final_kv_cache, fmt, skip_existing=True)
    t2 = time.perf_counter()
    engine2.store(final_tokens, final_kv_cache, fmt, skip_existing=False)
    t3 = time.perf_counter()

    print("With skip:", t2 - t1)
    print("No skip:", t3 - t2)
