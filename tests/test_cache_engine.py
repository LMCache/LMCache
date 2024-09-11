import pytest
import time
import os
import torch
import subprocess
import shlex

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder

def dumb_metadata(fmt="vllm"):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt)

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
        right_k = left_k.to(left_k.device)
        right_v = left_v.to(left_v.device)

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

def check_kv_cache_device(kvs, device):
    for kv in kvs:
        k, v = kv
        assert k.device == torch.device(device)
        assert v.device == torch.device(device)
        

# TODO(Jiayi): this test needs to be improved onece more dst_device is supported
@pytest.mark.parametrize("src_device", ["cuda:0", "cuda", "cpu"])
@pytest.mark.parametrize("dst_device", ["cuda:0"])
@pytest.mark.parametrize("backend", ["cuda", "cpu", "file://local_disk/"])
def test_retrive_device(backend, src_device, dst_device, autorelease):
    
    fmt = 'vllm'
    num_tokens = 500

    ''' initialize the engine '''
    tokens = generate_tokens(num_tokens, src_device)
    kv_cache = generate_kv_cache(num_tokens, fmt, src_device)
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = 256, backend = backend)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
    
    engine.store(tokens, kv_cache)
    retrived_cache, length = engine.retrive(tokens)
    check_kv_cache_device(retrived_cache, dst_device)
    

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("backend", ["cuda", "cpu", "file://local_disk/", "redis://localhost:6379", "lm://localhost:65000"])
@pytest.mark.parametrize("remote_serde", ["torch", "safetensor"]) # lossless serde
#@pytest.mark.usefixtures("lmserver_process")
@pytest.mark.parametrize("lmserver_process", ["cpu", "remote_disk/"], indirect=True)
def test_same_retrive_store(fmt, backend, remote_serde, autorelease, lmserver_process):
    device = "cpu" if backend == "cpu" else "cuda"
    num_tokens = 2000

    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = 256, backend = backend, remote_serde = remote_serde)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    ''' test retrive empty '''
    retrived_cache, length = engine.retrive(tokens)
    assert len(retrived_cache) == 0
    assert length == 0

    ''' test store '''
    engine.store(tokens, kv_cache)

    ''' test retrive '''
    retrived_cache, length = engine.retrive(tokens)

    assert length == num_tokens
    check_kv_cache_equal(retrived_cache, kv_cache, num_tokens, fmt)
    
    '''erase local cache'''
    if backend in ["file://local_disk/"]:
        subprocess.run(shlex.split(f"rm -rf local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cuda", "cpu", "file://local_disk/", "redis://localhost:6379", "lm://localhost:65000"])
@pytest.mark.parametrize("lmserver_process", ["cpu"], indirect=True)
def test_retrive_prefix(fmt, chunk_size, backend, autorelease, lmserver_process):
    device = "cpu" if backend == "cpu" else "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    print(fmt, chunk_size, backend)
    t1 = time.perf_counter()
    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    t2 = time.perf_counter()
    print(f"init tensor takes {t2-t1}")
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = chunk_size, backend=backend)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
    t3 = time.perf_counter()
    print(f"init engine takes {t3-t2}")
    
    ''' test store '''
    engine.store(tokens, kv_cache)
    t4 = time.perf_counter()
    print(f"store takes {t4-t3}")
    
    ''' test retrive '''
    retrived_cache, length = engine.retrive(torch.cat([tokens, new_tokens]))
    t5 = time.perf_counter()
    print(f"retrieve takes {t5-t4}")
    
    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrived_cache, kv_cache, expected_length, fmt)
    
    if backend in ["file://local_disk/"]:
        subprocess.run(shlex.split(f"rm -rf local_disk/"))


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cuda", "redis://localhost:6379", "lm://localhost:65000"])
@pytest.mark.parametrize("lmserver_process", ["cpu"], indirect=True)
def test_mixed_retrive(fmt, chunk_size, backend, autorelease, lmserver_process):
    device = "cuda"
    num_tokens = 2000
    new_num_tokens = 1000
    print(fmt, chunk_size, backend)
    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache(num_tokens, fmt, device)
    new_tokens = generate_tokens(new_num_tokens, device)
    new_kv_cache = generate_kv_cache(new_num_tokens, fmt, device)
    
    ''' initialize the engine '''
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = chunk_size, backend = backend)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    ''' test store '''
    engine.store(tokens, kv_cache)
    engine.store(new_tokens, new_kv_cache)

    ''' test retrive '''
    retrived_cache, length = engine.retrive(torch.cat([tokens, new_tokens]))

    expected_chunk_cnt = num_tokens // chunk_size
    expected_length = expected_chunk_cnt * chunk_size
    assert length == expected_length
    check_kv_cache_equal(retrived_cache, kv_cache, expected_length, fmt)
    
    ''' test another retrive '''
    retrived_cache, length = engine.retrive(new_tokens)

    assert length == new_num_tokens
    check_kv_cache_equal(retrived_cache, new_kv_cache, length, fmt)

    ''' insert the mixed kv cache '''
    final_tokens = torch.cat([tokens, new_tokens])
    final_kv_cache = concatenate_kv_caches([kv_cache, generate_kv_cache(new_num_tokens, fmt, device)], fmt)
    engine.store(final_tokens, final_kv_cache)

    ''' should retrive the mixed version '''
    retrived_cache, length = engine.retrive(final_tokens)
    assert length == num_tokens + new_num_tokens
    check_kv_cache_equal(retrived_cache, final_kv_cache, length, fmt)
    
    '''destroy local disk path'''
    if backend in ["file://local_disk/"]:
        subprocess.run(shlex.split(f"rm -rf local_disk/"))

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_skipping(fmt, autorelease):
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
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = chunk_size, persist_path = persist_path)
    engine1 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))
    engine2 = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    ''' store and persist '''
    engine1.store(tokens, kv_cache)
    engine2.store(tokens, kv_cache)

    ''' store final '''
    t1 = time.perf_counter()
    engine1.store(final_tokens, final_kv_cache, skip_existing=True)
    t2 = time.perf_counter()
    engine2.store(final_tokens, final_kv_cache, skip_existing=False)
    t3 = time.perf_counter()

    print("With skip:", t2 - t1)
    print("No skip:", t3 - t2)

def test_builder(autorelease):
    instance_id = "test"
    cfg = LMCacheEngineConfig.from_legacy(chunk_size = 256, persist_path = "/tmp/a.txt")
    cfg2 = LMCacheEngineConfig.from_legacy(chunk_size = 512, persist_path = "/tmp/a.txt")
    should_be_none = LMCacheEngineBuilder.get(instance_id)
    assert should_be_none is None

    engine = autorelease(LMCacheEngineBuilder.get_or_create(instance_id, cfg, dumb_metadata()))
    engine2 = autorelease(LMCacheEngineBuilder.get(instance_id))

    with pytest.raises(ValueError):
        LMCacheEngineBuilder.get_or_create(instance_id, cfg2, dumb_metadata())
