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

def generate_kv_cache(num_tokens, fmt, device, fill = None):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size] if fmt == "vllm" else [num_heads, num_tokens, head_size]
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(32):
        k = torch.rand(shape, dtype = dtype, device = device)
        v = torch.rand(shape, dtype = dtype, device = device)
        if fill is not None:
            k.fill_(fill)
            v.fill_(fill)
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

def slice_kv_caches(kv_chunk, s: slice, fmt):
    ret = []
    for kv_layer in kv_chunk:
        k, v = kv_layer
        kslice = k[s, ...] if fmt == "vllm" else k[:, s, ...]
        vslice = v[s, ...] if fmt == "vllm" else v[:, s, ...]
        ret.append((kslice.detach().clone(), vslice.detach().clone()))
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
    v_layer = kv_tuple[layer_id][1]

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
    kvs = [generate_kv_cache(length, fmt, "cuda", fill = None) for idx, length in enumerate(chunk_lengths)]
    tokens = [generate_tokens_with_spt(length, spt.device, spt) for length in chunk_lengths]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for token, kv in zip(tokens, kvs):
        engine.store(token, kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    def check_groups(*ids):
        input1 = torch.cat([tokens[i] for i in ids])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        target_len = sum([chunk_lengths[i] for i in ids])
        ret = retriever.new_request(input1, torch.tensor([0, target_len]).to(torch.int))
        for layer_id in range(32):
            result = ret.result(layer_id)
            check_kv_layer_equal(target_kv, layer_id, result.k, result.v, 0, target_len, fmt)
            assert (result.valid_mask == 1).all(), "Should be all valid!"

    check_groups(0)
    check_groups(0, 1)
    check_groups(0, 2)
    check_groups(0, 1, 2, 3)
    check_groups(1, 1, 2, 2)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("spt_length", [1, 2])
def test_spt_hit_miss(fmt, spt_length, autorelease):
    """
    This test tests the following use cases:
    - Some chunks are completely missing, some chunks are fully hit
    """

    # generate special tokens
    spt = generate_spt(spt_length)
    
    chunk_lengths = [1000, 2000, 1500, 3000]
    has_insterted = [True, False, True, False]
    kvs = [generate_kv_cache(length, fmt, "cuda", fill = None) for idx, length in enumerate(chunk_lengths)]
    tokens = [generate_tokens_with_spt(length, spt.device, spt) for length in chunk_lengths]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for flag, token, kv in zip(has_insterted, tokens, kvs):
        if flag:
            engine.store(token, kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    def check_groups(*ids):
        input1 = torch.cat([tokens[i] for i in ids])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        target_len = sum([chunk_lengths[i] for i in ids])
        ret = retriever.new_request(input1, torch.tensor([0, target_len]).to(torch.int))
        for layer_id in range(32):
            result = ret.result(layer_id)
            start_token = 0
            for i in ids:
                chunk_len = chunk_lengths[i]
                if has_insterted[i]:
                    check_kv_layer_equal(target_kv, layer_id, 
                                         result.k, result.v, start_token, start_token + chunk_len, fmt)
                    assert (result.valid_mask[start_token:start_token + chunk_len] == 1).all()
                else:
                    assert (result.valid_mask[start_token:start_token + chunk_len] == 0).all()
                start_token += chunk_len

    check_groups(0, 1, 2) # Y, N, Y
    check_groups(1, 2, 3) # N, Y, N

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("spt_length", [1, 2])
def test_spt_all_miss(fmt, spt_length, autorelease):
    """
    This test tests the following use cases:
    - All the chunks are completely missing
    """

    # generate special tokens
    spt = generate_spt(spt_length)
    
    chunk_lengths = [1000, 2000, 1500, 3000]
    has_insterted = [False, False, False, False]
    kvs = [generate_kv_cache(length, fmt, "cuda", fill = None) for idx, length in enumerate(chunk_lengths)]
    tokens = [generate_tokens_with_spt(length, spt.device, spt) for length in chunk_lengths]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for flag, token, kv in zip(has_insterted, tokens, kvs):
        if flag:
            engine.store(token, kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    def check_groups(*ids):
        input1 = torch.cat([tokens[i] for i in ids])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        target_len = sum([chunk_lengths[i] for i in ids])
        ret = retriever.new_request(input1, torch.tensor([0, target_len]).to(torch.int))
        for layer_id in range(32):
            result = ret.result(layer_id)
            assert result.k is None
            assert result.v is None
            assert (result.valid_mask == 0).all()

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("spt_length", [1, 2])
def test_spt_partial_hit(fmt, spt_length, autorelease):
    """
    This test tests the following use cases:
    - Partially hit chunks
    """

    # generate special tokens
    spt = generate_spt(spt_length)
    
    chunk_lengths = [1000, 2000, 1500, 3000]
    inserted_length = [500, 1000, 800, 1250]
    kvs = [generate_kv_cache(length, fmt, "cuda", fill = None) for idx, length in enumerate(chunk_lengths)]
    tokens = [generate_tokens_with_spt(length, spt.device, spt) for length in chunk_lengths]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for ilen, token, kv in zip(inserted_length, tokens, kvs):
        s = slice(0, ilen)
        partial_kv = slice_kv_caches(kv, s, fmt)
        partial_token = token[s]
        engine.store(partial_token, partial_kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    def check_groups(*ids):
        input1 = torch.cat([tokens[i] for i in ids])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        target_len = sum([chunk_lengths[i] for i in ids])
        ret = retriever.new_request(input1, torch.tensor([0, target_len]).to(torch.int))
        for layer_id in range(32):
            result = ret.result(layer_id)
            start_token = 0
            for i in ids:
                chunk_len = chunk_lengths[i]
                matched_len = result.valid_mask[start_token:start_token+chunk_len].sum()

                check_kv_layer_equal(target_kv, layer_id,
                                     result.k, result.v,
                                     start_token, start_token + matched_len, fmt)
                assert (result.valid_mask[start_token:start_token+matched_len] == 1).all()
                assert (result.valid_mask[start_token+matched_len:start_token+chunk_len] == 0).all()

                start_token += chunk_len

    check_groups(0)
    check_groups(0, 1)
    check_groups(0, 1, 2, 3)
    check_groups(0, 0)

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("spt_length", [1, 2])
def test_spt_multi_query(fmt, spt_length, autorelease):
    """
    This test tests the following use cases:
    - Have multiple queries in a batch, need to split at the query boundary 
    even if there is no spt
    """
    # generate special tokens
    spt = generate_spt(spt_length)
    
    chunk_lengths = [1000, 2000, 1500, 3000]
    kvs = [generate_kv_cache(length, fmt, "cuda", fill = None) for idx, length in enumerate(chunk_lengths)]
    tokens = [generate_tokens(length, "cpu") for length in chunk_lengths]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for token, kv in zip(tokens, kvs):
        engine.store(token, kv)

    retriever = SPTBlendRetriever(spt, engine, metadata)

    def check_groups(*ids):
        input1 = torch.cat([tokens[i] for i in ids])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        query_start_locs = [0]
        for i in ids:
            last = query_start_locs[-1]
            query_start_locs.append(last + chunk_lengths[i])

        ret1 = retriever.new_request(input1, torch.tensor(query_start_locs).to(torch.int))
        ret2 = retriever.new_request(input1, torch.tensor([0, query_start_locs[-1]]).to(torch.int))

        target_len1 = query_start_locs[-1]
        target_len2 = int(query_start_locs[1] // 256) * 256

        for layer_id in range(32):
            result1 = ret1.result(layer_id)
            check_kv_layer_equal(target_kv, layer_id, 
                                 result1.k, result1.v, 
                                 0, target_len1, fmt)
            assert (result1.valid_mask == 1).all(), "Should be all valid!"

            # Only the first chunk should be retrieved if there is no "query_start_loc"
            result2 = ret2.result(layer_id)
            check_kv_layer_equal(target_kv, layer_id, 
                                 result2.k, result2.v, 
                                 0, target_len2, fmt)
            assert (result2.valid_mask[0:target_len2] == 1).all(), "Should be all valid!"
            assert (result2.valid_mask[target_len2:] == 0).all(), "Should be all valid!"


    check_groups(0, 1)
    check_groups(0, 2)
    check_groups(0, 1, 2, 3)
    check_groups(1, 1, 2, 2)
