# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.blend.executor import CacheBlendImpl
from lmcache.blend.retriever import SPTBlendRetriever
from lmcache.cache_engine import LMCacheEngine
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata


def dumb_metadata(fmt="vllm", kv_shape=(32, 2, 256, 8, 128)):
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16
    return LMCacheEngineMetadata("test_model", 3, 123, fmt, dtype, kv_shape)


def dumb_cfg():
    return LMCacheEngineConfig.from_defaults(
        local_device="cuda",
        remote_url=None,
        remote_serde=None,
        enable_blending=True,
    )


def generate_kv_cache(num_tokens, fmt, device, fill=None):
    assert num_tokens >= 0
    ret = []
    num_heads = 8
    head_size = 128
    shape = (
        [num_tokens, num_heads, head_size]
        if fmt == "vllm"
        else [num_heads, num_tokens, head_size]
    )
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(32):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        if fill is not None:
            k.fill_(fill)
            v.fill_(fill)
        ret.append((k, v))

    return tuple(ret)


def fake_encode(text: str):
    return [int(token) for token in text.split()]


def fake_decode(token_ids: List[int]):
    return " ".join([str(token) for token in token_ids])


def generate_text(num_tokens) -> str:
    return fake_decode(random.choices(range(10000), k=num_tokens))


def get_spt():
    return "[BLEND_SEP]"


def drop_encode_and_indices(prompt: str, spt: str) -> Tuple[List[int], List[int]]:
    text_chunk_list = prompt.split(spt)
    input_ids = []
    input_text = ""
    blend_indices = []
    current_idx = 0
    for text_chunk in text_chunk_list:
        encoded = fake_encode(text_chunk)
        input_ids.extend(encoded)
        input_text += text_chunk
        current_idx += len(encoded)
        blend_indices.append(current_idx)
    if len(blend_indices) > 0:
        blend_indices.pop()
    return input_ids, blend_indices


def concatenate_kv_caches(kv_chunks, fmt):
    dim = 1 if fmt == "huggingface" else 0
    ret = []
    for kv_layer in zip(*kv_chunks, strict=False):
        klist, vlist = zip(*kv_layer, strict=False)
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


def check_has_spt(tokens, spt):
    """
    Check if the tokens have the spt
    """
    assert len(spt) > 0
    if len(tokens) < len(spt):
        return False
    else:
        i = 0
        while True:
            endi = i + len(spt)
            if endi > len(tokens):
                break
            if tokens[i:endi] == spt:
                return True
            i += 1
        return False


def assert_indices_is_concat(indices, chunk_lengths):
    """
    Check if the indices is the concatenation of the chunk lengths
    """
    assert len(indices) == len(chunk_lengths) - 1
    if len(indices) == 0:
        return
    this_seg_start = 0
    for i, idx in enumerate(indices):
        assert idx >= this_seg_start
        seg_len = idx - this_seg_start
        assert seg_len == chunk_lengths[i]
        this_seg_start = idx
    assert len(chunk_lengths) >= 2
    assert indices[-1] == this_seg_start
    assert indices[-1] == sum(chunk_lengths[:-1])


@pytest.mark.parametrize("fmt", ["vllm"])
def test_spt_full_hit(fmt, autorelease):
    """
    This test tests the following use cases:
    - All chunks are fully hit
    - Some chunks are completely missing, some chunks are fully hit
    - Chunks are partially hit
    - No chunks are hit
    """

    # generate special tokens
    spt = get_spt()

    chunk_lengths = [1000, 2000, 1500, 3000]
    kvs = [
        generate_kv_cache(length, fmt, "cuda", fill=None)
        for idx, length in enumerate(chunk_lengths)
    ]
    tokens = [generate_text(length) for length in chunk_lengths]

    token_ids = [fake_encode(token) for token in tokens]
    token_ids_tensors = [torch.tensor(token_id, device="cpu") for token_id in token_ids]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for token_ids_tensor, kv in zip(token_ids_tensors, kvs, strict=False):
        engine.store(token_ids_tensor, kv)

    retriever = SPTBlendRetriever(engine, metadata)

    def check_groups(*ids):
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        query_prompt = spt.join([tokens[i] for i in ids])
        input_ids, blend_indices = drop_encode_and_indices(query_prompt, spt)
        new_prompt = torch.tensor(input_ids, device="cpu")
        ret = retriever.new_request([new_prompt], [blend_indices])
        target_len = sum([chunk_lengths[i] for i in ids])
        for layer_id in range(32):
            result = ret.result(layer_id)
            check_kv_layer_equal(
                target_kv, layer_id, result.k, result.v, 0, target_len, fmt
            )
            assert (result.valid_mask == 1).all(), "Should be all valid!"
            gt_positions = torch.cat([torch.arange(chunk_lengths[i]) for i in ids])
            assert (result.original_positions == gt_positions).all()

    check_groups(0)
    check_groups(0, 1)
    check_groups(0, 2)
    check_groups(0, 1, 2, 3)
    check_groups(1, 1, 2, 2)


@pytest.mark.parametrize("fmt", ["vllm"])
def test_spt_hit_miss(fmt, autorelease):
    """
    This test tests the following use cases:
    - Some chunks are completely missing, some chunks are fully hit
    """

    # generate special tokens
    spt = get_spt()

    chunk_lengths = [1000, 2000, 1500, 3000]
    has_insterted = [True, False, True, False]
    kvs = [
        generate_kv_cache(length, fmt, "cuda", fill=None)
        for idx, length in enumerate(chunk_lengths)
    ]
    tokens = [generate_text(length) for length in chunk_lengths]
    token_ids = [fake_encode(token) for token in tokens]
    token_ids_tensors = [torch.tensor(token_id, device="cpu") for token_id in token_ids]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for flag, token_ids_tensor, kv in zip(
        has_insterted, token_ids_tensors, kvs, strict=False
    ):
        if flag:
            engine.store(token_ids_tensor, kv)

    retriever = SPTBlendRetriever(engine, metadata)

    def check_groups(*ids):
        query_prompt = spt.join([tokens[i] for i in ids])
        input_ids, blend_indices = drop_encode_and_indices(query_prompt, spt)
        new_prompt = torch.tensor(input_ids, device="cpu")
        ret = retriever.new_request([new_prompt], [blend_indices])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        for layer_id in range(32):
            result = ret.result(layer_id)
            start_token = 0
            for i in ids:
                chunk_len = chunk_lengths[i]
                assert chunk_len >= 0
                if has_insterted[i]:
                    check_kv_layer_equal(
                        target_kv,
                        layer_id,
                        result.k,
                        result.v,
                        start_token,
                        start_token + chunk_len,
                        fmt,
                    )
                    assert (
                        result.valid_mask[start_token : start_token + chunk_len] == 1
                    ).all()
                    gt_positions = torch.arange(chunk_len)
                    assert (
                        result.original_positions[start_token : start_token + chunk_len]
                        == gt_positions
                    ).all()
                else:
                    assert (
                        result.valid_mask[start_token : start_token + chunk_len] == 0
                    ).all()
                    assert (
                        result.original_positions[start_token : start_token + chunk_len]
                        == 0
                    ).all()
                start_token += chunk_len

    check_groups(0, 1, 2)  # Y, N, Y
    check_groups(1, 2, 3)  # N, Y, N


@pytest.mark.parametrize("fmt", ["vllm"])
def test_spt_all_miss(fmt, autorelease):
    """
    This test tests the following use cases:
    - All the chunks are completely missing
    """

    # generate special tokens
    spt = get_spt()

    chunk_lengths = [1000, 2000, 1500, 3000]
    has_insterted = [False, False, False, False]
    kvs = [
        generate_kv_cache(length, fmt, "cuda", fill=None)
        for idx, length in enumerate(chunk_lengths)
    ]
    tokens = [generate_text(length) for length in chunk_lengths]
    token_ids = [fake_encode(token) for token in tokens]
    token_ids_tensors = [torch.tensor(token_id, device="cpu") for token_id in token_ids]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for flag, token_ids_tensor, kv in zip(
        has_insterted, token_ids_tensors, kvs, strict=False
    ):
        if flag:
            engine.store(token_ids_tensor, kv)

    retriever = SPTBlendRetriever(engine, metadata)

    def check_groups(*ids):
        query_prompt = spt.join([tokens[i] for i in ids])
        input_ids, blend_indices = drop_encode_and_indices(query_prompt, spt)
        new_prompt = torch.tensor(input_ids, device="cpu")
        ret = retriever.new_request([new_prompt], [blend_indices])
        for layer_id in range(32):
            result = ret.result(layer_id)
            assert result.k is None
            assert result.v is None
            assert (result.valid_mask == 0).all()
            assert (result.original_positions == 0).all()

    check_groups(0, 1, 2, 3)
    check_groups(1, 2, 3)


@pytest.mark.parametrize("fmt", ["vllm"])
def test_spt_partial_hit(fmt, autorelease):
    """
    This test tests the following use cases:
    - Partially hit chunks
    """

    # generate special tokens
    spt = get_spt()

    chunk_lengths = [1000, 2000, 1500, 3000]
    inserted_length = [500, 1000, 800, 1250]
    kvs = [
        generate_kv_cache(length, fmt, "cuda", fill=None)
        for idx, length in enumerate(chunk_lengths)
    ]
    tokens = [generate_text(length) for length in chunk_lengths]
    token_ids = [fake_encode(token) for token in tokens]
    token_ids_tensors = [torch.tensor(token_id, device="cpu") for token_id in token_ids]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for ilen, token_ids_tensor, kv in zip(
        inserted_length, token_ids_tensors, kvs, strict=False
    ):
        assert ilen < len(token_ids_tensor)
        s = slice(0, ilen)
        partial_kv = slice_kv_caches(kv, s, fmt)
        partial_token_ids_tensor = token_ids_tensor[s]
        engine.store(partial_token_ids_tensor, partial_kv)

    retriever = SPTBlendRetriever(engine, metadata)

    def check_groups(*ids):
        query_prompt = spt.join([tokens[i] for i in ids])
        input_ids, blend_indices = drop_encode_and_indices(query_prompt, spt)
        new_prompt = torch.tensor(input_ids, device="cpu")
        ret = retriever.new_request([new_prompt], [blend_indices])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        for layer_id in range(32):
            result = ret.result(layer_id)
            start_token = 0
            for i in ids:
                chunk_len = chunk_lengths[i]
                matched_len = result.valid_mask[
                    start_token : start_token + chunk_len
                ].sum()

                check_kv_layer_equal(
                    target_kv,
                    layer_id,
                    result.k,
                    result.v,
                    start_token,
                    start_token + matched_len,
                    fmt,
                )
                assert (
                    result.valid_mask[start_token : start_token + matched_len] == 1
                ).all()
                assert (
                    result.valid_mask[
                        start_token + matched_len : start_token + chunk_len
                    ]
                    == 0
                ).all()

                gt_positions = torch.arange(matched_len)
                assert (
                    result.original_positions[start_token : start_token + matched_len]
                    == gt_positions
                ).all()
                assert (
                    result.original_positions[
                        start_token + matched_len : start_token + chunk_len
                    ]
                    == 0
                ).all()

                start_token += chunk_len

    check_groups(0)
    check_groups(0, 1)
    check_groups(0, 1, 2, 3)
    check_groups(0, 0)


@pytest.mark.parametrize("fmt", ["vllm"])
def test_spt_multi_query(fmt, autorelease):
    """
    This test tests the following use cases:
    - Have multiple queries in a batch, need to split at the query boundary
    even if there is no spt
    """

    chunk_lengths = [1000, 2000, 1500, 3000]
    kvs = [
        generate_kv_cache(length, fmt, "cuda", fill=None)
        for idx, length in enumerate(chunk_lengths)
    ]
    tokens = [generate_text(length) for length in chunk_lengths]
    token_ids = [fake_encode(token) for token in tokens]
    token_ids_tensors = [torch.tensor(token_id, device="cpu") for token_id in token_ids]

    cfg = dumb_cfg()
    metadata = dumb_metadata(fmt)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    for token_ids_tensor, kv in zip(token_ids_tensors, kvs, strict=False):
        engine.store(token_ids_tensor, kv)

    retriever = SPTBlendRetriever(engine, metadata)

    def check_groups(*ids) -> None:
        query_prompt_list = [tokens[i] for i in ids]
        input_ids_list = []
        blend_indices_list: List[List[int]] = []
        for query_prompt in query_prompt_list:
            input_ids = fake_encode(query_prompt)
            input_ids_list.append(torch.tensor(input_ids, device="cpu"))
            blend_indices_list.append([])
        target_kv = concatenate_kv_caches([kvs[i] for i in ids], fmt)
        ret1 = retriever.new_request(input_ids_list, blend_indices_list)
        single_prompt = " ".join(query_prompt_list)
        single_prompt_tensor = torch.tensor(fake_encode(single_prompt), device="cpu")
        ret2 = retriever.new_request([single_prompt_tensor], [[]])
        target_len1 = sum([chunk_lengths[i] for i in ids])
        # NOTE: Assuming chunk size is 256.
        target_len2 = int(chunk_lengths[ids[0]] // 256) * 256

        for layer_id in range(32):
            result1 = ret1.result(layer_id)
            check_kv_layer_equal(
                target_kv, layer_id, result1.k, result1.v, 0, target_len1, fmt
            )
            assert (result1.valid_mask == 1).all(), "Should be all valid!"

            # Only the first chunk should be retrieved if there is no
            # "query_start_loc"
            result2 = ret2.result(layer_id)
            check_kv_layer_equal(
                target_kv, layer_id, result2.k, result2.v, 0, target_len2, fmt
            )
            assert (result2.valid_mask[0:target_len2] == 1).all(), (
                "Should be all valid!"
            )
            assert (result2.valid_mask[target_len2:] == 0).all(), (
                "Should be all invalid!"
            )

    check_groups(0, 1)
    check_groups(0, 2)
    check_groups(0, 1, 2, 3)
    check_groups(1, 1, 2, 2)


def test_cacheblend_executor_single_query():
    # Case 1: all valid
    dtype = torch.bfloat16
    device = "cuda"
    prefix_len = 10
    query_len = 10
    q_shape = (query_len, 4096)
    kv_shape = (query_len, 1024)

    changed_positions = [2, 6]
    expected_positions = [p + prefix_len for p in changed_positions]

    def dumb_posional_encoding(p, q, k):
        return q, k

    blender = CacheBlendImpl(0.2)
    blender.set_positional_encoder(dumb_posional_encoding)
    blender.set_reverse_positional_encoder(dumb_posional_encoding)

    fq_1 = torch.zeros(q_shape, dtype=dtype, device=device)
    for i in range(query_len):
        fq_1[i] = i

    # Newly generated KV is 0 on the "changed_positions"
    fk_1 = torch.full(kv_shape, 1, dtype=dtype, device=device)
    fk_1[changed_positions, ...] = 0
    fv_1 = torch.full(kv_shape, 1, dtype=dtype, device=device)
    fv_1[changed_positions, ...] = 0

    # Retrieved KV are all 1
    rk_1 = torch.full(kv_shape, 1, dtype=dtype, device=device)
    rv_1 = torch.full(kv_shape, 1, dtype=dtype, device=device)
    valid = torch.full((query_len,), 1, dtype=torch.long, device="cpu")
    positions = torch.arange(
        prefix_len, prefix_len + query_len, dtype=torch.int32, device="cuda"
    )
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    original_positions = torch.arange(query_len)

    # First layer should do nothing!
    ret = blender.blend(
        0,
        rk_1,
        rv_1,
        valid,
        original_positions,
        fq_1,
        fk_1,
        fv_1,
        positions,
        query_start_loc,
        0,
    )
    assert torch.equal(ret.q, fq_1)
    assert torch.equal(ret.k, fk_1)
    assert torch.equal(ret.v, fv_1)
    assert torch.equal(ret.positions, positions)
    assert torch.equal(
        ret.local_indices,
        torch.arange(prefix_len, dtype=torch.int, device="cpu"),
    )
    assert ret.query_start_loc is None

    # Second layer should do token selection
    ret = blender.blend(
        1,
        rk_1,
        rv_1,
        valid,
        original_positions,
        fq_1,
        fk_1,
        fv_1,
        positions,
        query_start_loc,
        0,
    )
    assert len(ret.positions) == len(expected_positions)  # recompute 2 tokens
    assert ret.k.shape[0] == query_len  # long K
    assert ret.v.shape[0] == query_len  # long V
    assert torch.equal(
        ret.local_indices,
        torch.tensor(changed_positions, dtype=torch.int, device="cpu"),
    )
    assert ret.query_start_loc[0].item() == 0
    assert ret.query_start_loc[1].item() == 2
    for i in range(len(expected_positions)):
        assert ret.positions[i].item() == expected_positions[i]
        assert ret.q[i][0].item() == changed_positions[i]
        assert (ret.k[changed_positions[i]] == 0).all()
        assert (ret.v[changed_positions[i]] == 0).all()

    # Third layer should do kv update
    fq_2 = ret.q
    fk_2 = fk_1[changed_positions]
    fv_2 = fv_1[changed_positions]
    rk_2 = rk_1
    rv_2 = rv_1
    pos_2 = ret.positions
    ret = blender.blend(
        2,
        rk_2,
        rv_2,
        valid,
        original_positions,
        ret.q,
        fk_2,
        fv_2,
        pos_2,
        query_start_loc,
        0,
    )

    # Should update the KV without changing q or positions
    assert torch.equal(ret.q, fq_2)
    assert torch.equal(ret.positions, pos_2)
    assert ret.k.shape[0] == prefix_len
    assert ret.v.shape[0] == prefix_len
    assert (ret.k[changed_positions] == 0).all()
    assert (ret.v[changed_positions] == 0).all()
    unchanged_positions = list(
        filter(lambda x: x not in changed_positions, range(query_len))
    )
    assert (ret.k[unchanged_positions] == 1).all()
    assert (ret.v[unchanged_positions] == 1).all()
    assert torch.equal(
        ret.local_indices,
        torch.tensor(changed_positions, dtype=torch.int, device="cpu"),
    )
    assert ret.query_start_loc is None

    # TODO: un-tested cases:
    # - some positions are invalid
    # - multiple queries (batch size > 1)
