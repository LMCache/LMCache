import pytest
import torch
from lmcache.cache_engine import LMCacheEngine

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
    engine = LMCacheEngine(chunk_size = 256)

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
    engine = LMCacheEngine(chunk_size = chunk_size)

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
    engine = LMCacheEngine(chunk_size = chunk_size)

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
