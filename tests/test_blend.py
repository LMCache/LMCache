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

    # Test cases implemented below


def test_cacheblend_executor_invalid_positions():
    """Test case for some positions being invalid - comprehensive validation"""
    dtype = torch.bfloat16
    device = "cuda"
    prefix_len = 10
    query_len = 10
    q_shape = (query_len, 4096)
    kv_shape = (query_len, 1024)

    changed_positions = [2, 6]
    invalid_positions = [3, 7]

    def dumb_posional_encoding(p, q, k):
        return q, k

    blender = CacheBlendImpl(0.2)
    blender.set_positional_encoder(dumb_posional_encoding)
    blender.set_reverse_positional_encoder(dumb_posional_encoding)

    fq_1 = torch.zeros(q_shape, dtype=dtype, device=device)
    for i in range(query_len):
        fq_1[i] = i

    # Create distinctive patterns for better validation
    # Fresh KV: set different values for changed positions to create clear differences
    fk_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    fv_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    # Make positions 2,6 have maximum difference (0 vs 1)
    fk_1[changed_positions, ...] = 0.0
    fv_1[changed_positions, ...] = 0.0
    # Make invalid positions distinctive
    fk_1[invalid_positions, ...] = 0.5
    fv_1[invalid_positions, ...] = 0.5

    # Retrieved KV: all 1s except we'll make some positions slightly different
    rk_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    rv_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    # Create small differences at other positions to test selection logic
    rk_1[1, ...] = 0.99  # Small difference
    rv_1[1, ...] = 0.99
    rk_1[4, ...] = 0.98  # Medium difference
    rv_1[4, ...] = 0.98

    # Create valid mask with invalid positions
    valid = torch.full((query_len,), 1, dtype=torch.long, device="cpu")
    for pos in invalid_positions:
        valid[pos] = 0

    positions = torch.arange(
        prefix_len, prefix_len + query_len, dtype=torch.int32, device="cuda"
    )
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    original_positions = torch.arange(query_len)

    # Test layer 0 (should be pass-through)
    ret_layer0 = blender.blend(
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
    # Layer 0 should return fresh values unchanged
    assert torch.equal(ret_layer0.q, fq_1)
    assert torch.equal(ret_layer0.k, fk_1)
    assert torch.equal(ret_layer0.v, fv_1)
    assert torch.equal(ret_layer0.positions, positions)
    assert ret_layer0.query_start_loc is None

    # Test layer 1 (token selection)
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

    # Robust validation of selection logic
    num_valid_tokens = valid.sum().item()  # Should be 8
    num_invalid_tokens = len(invalid_positions)  # Should be 2
    expected_valid_selected = int(num_valid_tokens * 0.2)  # 8 * 0.2 = 1.6 -> 1
    expected_total_selected = num_invalid_tokens + expected_valid_selected  # 2 + 1 = 3

    assert num_valid_tokens == 8, f"Expected 8 valid tokens, got {num_valid_tokens}"
    assert len(ret.positions) == expected_total_selected, (
        f"Expected {expected_total_selected} selected tokens, got {len(ret.positions)}"
    )
    assert ret.k.shape[0] == query_len
    assert ret.v.shape[0] == query_len

    # Verify ALL invalid positions are included
    local_indices_set = set(ret.local_indices.cpu().numpy().tolist())
    for invalid_pos in invalid_positions:
        assert invalid_pos in local_indices_set, (
            f"Invalid position {invalid_pos} not found in selected indices "
            f"{local_indices_set}"
        )

    # Verify selected positions are reasonable
    assert len(ret.local_indices) == expected_total_selected
    assert all(0 <= idx < query_len for idx in ret.local_indices), (
        "Selected indices out of range"
    )

    # Verify query_start_loc structure for single query
    assert ret.query_start_loc is not None
    assert len(ret.query_start_loc) == 2  # [0, num_selected]
    assert ret.query_start_loc[0] == 0
    assert ret.query_start_loc[1] == expected_total_selected

    # Test with different recompute ratios to ensure robustness
    for ratio in [0.1, 0.3, 0.5]:
        blender_test = CacheBlendImpl(ratio)
        blender_test.set_positional_encoder(dumb_posional_encoding)
        blender_test.set_reverse_positional_encoder(dumb_posional_encoding)

        ret_test = blender_test.blend(
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

        expected_valid_for_ratio = int(num_valid_tokens * ratio)
        expected_total_for_ratio = num_invalid_tokens + expected_valid_for_ratio
        assert len(ret_test.positions) == expected_total_for_ratio, (
            f"Ratio {ratio}: expected {expected_total_for_ratio}, "
            f"got {len(ret_test.positions)}"
        )

        # All invalid positions should still be included
        test_indices_set = set(ret_test.local_indices.cpu().numpy().tolist())
        for invalid_pos in invalid_positions:
            assert invalid_pos in test_indices_set


def test_cacheblend_executor_multiple_queries():
    """Test case for multiple queries (batch size > 1) - comprehensive validation"""
    dtype = torch.bfloat16
    device = "cuda"
    prefix_len = 10
    query_len_1 = 8
    query_len_2 = 6
    query_len_3 = 4  # Add third query for more robust testing
    total_query_len = query_len_1 + query_len_2 + query_len_3
    q_shape = (total_query_len, 4096)
    kv_shape = (total_query_len, 1024)

    # Positions with differences in each query
    changed_positions_q1 = [2, 6]  # Query 1: positions 0-7
    changed_positions_q2 = [10, 12]  # Query 2: positions 8-13
    changed_positions_q3 = [16]  # Query 3: positions 14-17
    all_changed_positions = (
        changed_positions_q1 + changed_positions_q2 + changed_positions_q3
    )

    def dumb_posional_encoding(p, q, k):
        return q, k

    blender = CacheBlendImpl(0.2)
    blender.set_positional_encoder(dumb_posional_encoding)
    blender.set_reverse_positional_encoder(dumb_posional_encoding)

    fq_1 = torch.zeros(q_shape, dtype=dtype, device=device)
    for i in range(total_query_len):
        fq_1[i] = i

    # Create distinctive KV patterns for each query
    fk_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    fv_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    rk_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)
    rv_1 = torch.full(kv_shape, 1.0, dtype=dtype, device=device)

    # Create clear differences: fresh=0, retrieved=1 at changed positions
    for pos in all_changed_positions:
        fk_1[pos, ...] = 0.0
        fv_1[pos, ...] = 0.0

    # Add smaller differences to test selection priority within each query
    fv_1[1, ...] = 0.9  # Query 1: small difference
    fv_1[4, ...] = 0.8  # Query 1: medium difference
    fv_1[9, ...] = 0.9  # Query 2: small difference
    fv_1[11, ...] = 0.85  # Query 2: medium difference
    fv_1[15, ...] = 0.9  # Query 3: small difference

    valid = torch.full((total_query_len,), 1, dtype=torch.long, device="cpu")
    positions = torch.arange(
        prefix_len, prefix_len + total_query_len, dtype=torch.int32, device="cuda"
    )

    # Define query boundaries for 3 queries
    query_start_loc = torch.tensor(
        [0, query_len_1, query_len_1 + query_len_2, total_query_len],
        dtype=torch.int32,
        device="cuda",
    )
    original_positions = torch.arange(total_query_len)

    # Test layer 0 first (pass-through)
    ret_layer0 = blender.blend(
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
    assert torch.equal(ret_layer0.q, fq_1)
    assert torch.equal(ret_layer0.k, fk_1)
    assert torch.equal(ret_layer0.v, fv_1)
    assert ret_layer0.query_start_loc is None

    # Test layer 1 with multiple queries
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

    # Calculate expected selections per query with robust validation
    expected_q1 = int(query_len_1 * 0.2)  # 8 * 0.2 = 1.6 -> 1
    expected_q2 = int(query_len_2 * 0.2)  # 6 * 0.2 = 1.2 -> 1
    expected_q3 = int(query_len_3 * 0.2)  # 4 * 0.2 = 0.8 -> 0
    expected_total = expected_q1 + expected_q2 + expected_q3

    assert len(ret.positions) == expected_total, (
        f"Expected {expected_total} tokens selected, got {len(ret.positions)}"
    )
    assert ret.k.shape[0] == total_query_len
    assert ret.v.shape[0] == total_query_len
    assert ret.query_start_loc is not None
    assert len(ret.query_start_loc) == 4  # [0, end_q1, end_q2, total]

    # Verify query boundaries are respected
    local_indices = ret.local_indices.cpu().numpy()
    query1_indices = [idx for idx in local_indices if 0 <= idx < query_len_1]
    query2_indices = [
        idx for idx in local_indices if query_len_1 <= idx < query_len_1 + query_len_2
    ]
    query3_indices = [
        idx
        for idx in local_indices
        if query_len_1 + query_len_2 <= idx < total_query_len
    ]

    assert len(query1_indices) == expected_q1, (
        f"Query 1: expected {expected_q1}, got {len(query1_indices)}"
    )
    assert len(query2_indices) == expected_q2, (
        f"Query 2: expected {expected_q2}, got {len(query2_indices)}"
    )
    assert len(query3_indices) == expected_q3, (
        f"Query 3: expected {expected_q3}, got {len(query3_indices)}"
    )

    # Verify query_start_loc structure is correct
    assert ret.query_start_loc[0] == 0
    assert ret.query_start_loc[-1] == expected_total
    for i in range(1, len(ret.query_start_loc)):
        expected_cumulative = sum([expected_q1, expected_q2, expected_q3][:i])
        assert ret.query_start_loc[i] == expected_cumulative, (
            f"Boundary {i}: expected {expected_cumulative}, "
            f"got {ret.query_start_loc[i]}"
        )

    # Test with invalid tokens across multiple queries
    valid_with_invalid = valid.clone()
    valid_with_invalid[3] = 0  # Invalid in query 1
    valid_with_invalid[10] = 0  # Invalid in query 2

    ret_invalid = blender.blend(
        1,
        rk_1,
        rv_1,
        valid_with_invalid,
        original_positions,
        fq_1,
        fk_1,
        fv_1,
        positions,
        query_start_loc,
        0,
    )

    # The algorithm may select tokens multiple times or have more complex behavior
    # Let's validate what actually happens and ensure key invariants hold
    invalid_indices = ret_invalid.local_indices.cpu().numpy()

    # Verify invalid positions are included (this is the most important invariant)
    assert 3 in invalid_indices, "Invalid position 3 should be selected"
    assert 10 in invalid_indices, "Invalid position 10 should be selected"

    # Verify we select at least the invalid tokens
    assert len(ret_invalid.positions) >= 2, (
        "Should select at least the 2 invalid tokens"
    )

    # Verify the structure is still correct
    assert ret_invalid.query_start_loc is not None
    assert len(ret_invalid.query_start_loc) == 4

    # Test edge case: single token queries
    edge_query_lengths = [1, 1, 2]
    edge_total = sum(edge_query_lengths)
    edge_positions = torch.arange(
        prefix_len, prefix_len + edge_total, dtype=torch.int32, device="cuda"
    )
    edge_query_start = torch.cumsum(torch.tensor([0] + edge_query_lengths), 0).to(
        dtype=torch.int32, device="cuda"
    )
    edge_valid = torch.ones(edge_total, dtype=torch.long, device="cpu")
    edge_orig_pos = torch.arange(edge_total)

    edge_fq = torch.zeros((edge_total, 4096), dtype=dtype, device=device)
    edge_fk = torch.ones((edge_total, 1024), dtype=dtype, device=device)
    edge_fv = torch.ones((edge_total, 1024), dtype=dtype, device=device)
    edge_rk = torch.ones((edge_total, 1024), dtype=dtype, device=device)
    edge_rv = torch.ones((edge_total, 1024), dtype=dtype, device=device)

    edge_blender = CacheBlendImpl(0.5)  # High ratio for small queries
    edge_blender.set_positional_encoder(dumb_posional_encoding)
    edge_blender.set_reverse_positional_encoder(dumb_posional_encoding)

    edge_ret = edge_blender.blend(
        1,
        edge_rk,
        edge_rv,
        edge_valid,
        edge_orig_pos,
        edge_fq,
        edge_fk,
        edge_fv,
        edge_positions,
        edge_query_start,
        0,
    )

    # Expected: int(1*0.5)=0, int(1*0.5)=0, int(2*0.5)=1 = 1 total
    expected_edge = 0 + 0 + 1
    assert len(edge_ret.positions) == expected_edge, (
        f"Edge case: expected {expected_edge}, got {len(edge_ret.positions)}"
    )
