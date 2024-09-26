import pytest
import time
import os
import torch
import subprocess
import shlex

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.blend.retriever import SPTBlendRetriever

def dumb_metadata(fmt="vllm"):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt)

def dumb_cfg():
    return LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url = None, remote_serde = None)

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

def generate_spt(length):
    return torch.full((length, ), 10010)

def generate_tokens_with_spt(num_tokens, device, spt):
    """
    Generate the tokens ended with spt
    """
    minval = torch.min(spt).item()
    ret = torch.randint(minval - 100, minval - 10, size=[num_tokens]).to(device)
    ret[-len(spt):] = spt
    return ret

def concatenate_kv_caches(kv_chunks, fmt):
    dim = 1 if fmt == "huggingface" else 0
    ret = []
    for kv_layer in zip(*kv_chunks):
        klist, vlist = zip(*kv_layer)
        klayer = torch.cat(klist, dim=dim)
        vlayer = torch.cat(vlist, dim=dim)
        ret.append((klayer, vlayer))
    return tuple(ret)

def check_kv_cache_equal(left, right, start_token, end_token, fmt):
    """
    check if the first num_tokens of left and right kv cache are the same
    """
    dim = 0 if fmt == "vllm" else 1
    left_k = left
    right_k = right.to(left_k.device)

    assert len(left_k.shape) == 3
    assert len(right_k.shape) == 3

    s = slice(start_token, end_token)
    match fmt:
        case "huggingface":
            assert (left_k[:, s, :] == right_k[:, s, :]).all()
        case "vllm":
            assert (left_k[s, :, :] == right_k[s, :, :]).all()

def check_kv_layer_equal(kv_tuple, layer_id, k, v, start_token, end_token, fmt):
    k_layer = kv_tuple[layer_id][0]
    v_layer = kv_tuple[layer_id][0]

    check_kv_cache_equal(k_layer, k, start_token, end_token, fmt)
    check_kv_cache_equal(v_layer, v, start_token, end_token, fmt)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("spt_length", [1, 2])
def test_spt_full_hit(fmt, spt_length, autorelease):
    """
    This test tests the following use cases:
    - All chunks are fully hit
    - Some chunks are completely missing, some chunks are fully hit
    - Chunks are partially hit
    - No chunks are hit
    """


    # generate special tokens
    spt = generate_spt(spt_length)
    
    chunk_lengths = [1000, 2000, 1500, 3000]
    kvs = [generate_kv_cache(length, fmt, "cuda") for length in chunk_lengths]
    tokens = [generate_tokens_with_spt(length, spt.device, spt) for length in chunk_lengths]
    print(kvs[0][0][0].shape)
    print(tokens[0].shape)

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for token, kv in zip(tokens, kvs):
        engine.store(token, kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    input1 = torch.cat([tokens[0], tokens[2]])
    target_kv = concatenate_kv_caches([kvs[0], kvs[2]], fmt)
    ret = retriever.new_request(input1, torch.tensor([0]).to(torch.int))
    for layer_id in range(32):
        result = ret.result(layer_id)
        check_kv_layer_equal(target_kv, layer_id, result.k, result.v, 0, 2500, fmt)
        assert (result.valid_mask == 1).all(), "Should be all valid!"

    #input2 = torch.cat(tokens[1], tokens[3])
    #input3 = torch.cat(tokens[1], tokens[1])
    #input4 = torch.cat(tokens)

