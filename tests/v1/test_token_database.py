# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_tuple = tuple(sys_tokens.cpu().tolist())
    sys_hash = hash((None, sys_tuple, None))
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_tuple = tuple(token_chunk.cpu().tolist())
        token_hash = hash((None, token_tuple, None))
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_tuple = tuple(query_tokens.cpu().tolist())
    query_hash = hash((None, query_tuple, None))
    hashes.append(query_hash)
    starts.append(start)
    ends.append(start + query_length)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        # print(st, starts[i])
        # print(ed, ends[i])


@pytest.mark.parametrize(
    "hash_algorithm,extra_keys_val",
    [
        ("builtin", None),
        ("builtin", []),
        ("builtin", (42,)),
        ("sha256", None),
        ("sha256_cbor", None),
    ],
)
def test_hash_tokens_deterministicity(hash_algorithm, extra_keys_val):
    """Test that _hash_tokens produces deterministic results."""
    # Check if vLLM is available for sha256/sha256_cbor
    os.environ["PYTHONHASHSEED"] = "0"
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=256, backend="cpu", pre_caching_hash_algorithm=hash_algorithm
    )
    metadata = dumb_metadata()
    db = ChunkedTokenDatabase(cfg, metadata)

    # Test with torch.Tensor
    tokens_tensor = torch.tensor([1, 2, 3, 4, 5], device="cpu")
    prefix_hash = 12345
    extra_keys = extra_keys_val

    # Call multiple times - should produce same hash
    hash1 = db._hash_tokens(tokens_tensor, prefix_hash, extra_keys)
    hash2 = db._hash_tokens(tokens_tensor, prefix_hash, extra_keys)
    hash3 = db._hash_tokens(tokens_tensor, prefix_hash, extra_keys)

    assert hash1 == hash2 == hash3, "Hash should be deterministic"

    # Test with list - should produce same hash as equivalent tensor
    tokens_list = [1, 2, 3, 4, 5]
    hash_list = db._hash_tokens(tokens_list, prefix_hash, extra_keys)
    assert hash1 == hash_list, "List and tensor should produce same hash"


def test_hash_tokens_edge_cases():
    """Test edge cases for _hash_tokens."""
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
    metadata = dumb_metadata()
    db = ChunkedTokenDatabase(cfg, metadata)

    tokens = torch.tensor([1, 2, 3], device="cpu")

    # Test 1: extra_keys is None
    hash_none_extra = db._hash_tokens(tokens, prefix_hash=100, extra_keys=None)
    hash_empty_extra = db._hash_tokens(tokens, prefix_hash=100, extra_keys=[])
    # None and empty list should produce same hash (both converted to empty tuple)
    assert hash_none_extra == hash_empty_extra

    # Test 2: prefix_hash is None
    hash_none_prefix = db._hash_tokens(tokens, prefix_hash=None, extra_keys=None)
    hash_zero_prefix = db._hash_tokens(tokens, prefix_hash=0, extra_keys=None)
    # None and 0 should produce same hash (None converted to 0)
    assert hash_none_prefix == hash_zero_prefix

    # Test 3: Both None
    hash_both_none = db._hash_tokens(tokens, prefix_hash=None, extra_keys=None)
    hash_both_zero = db._hash_tokens(tokens, prefix_hash=0, extra_keys=[])
    assert hash_both_none == hash_both_zero

    # Test 4: extra_keys with values
    hash_with_extra = db._hash_tokens(
        tokens, prefix_hash=100, extra_keys=["key1", "key2", 123]
    )
    hash_without_extra = db._hash_tokens(tokens, prefix_hash=100, extra_keys=None)
    # Should be different when extra_keys is provided
    assert hash_with_extra != hash_without_extra

    # Test 5: Different prefix_hash produces different hash
    hash_prefix1 = db._hash_tokens(tokens, prefix_hash=100, extra_keys=None)
    hash_prefix2 = db._hash_tokens(tokens, prefix_hash=200, extra_keys=None)
    assert hash_prefix1 != hash_prefix2

    # Test 6: Empty tokens
    empty_tokens_tensor = torch.tensor([], device="cpu")
    empty_tokens_list = []
    hash_empty_tensor = db._hash_tokens(empty_tokens_tensor, prefix_hash=100)
    hash_empty_list = db._hash_tokens(empty_tokens_list, prefix_hash=100)
    assert hash_empty_tensor == hash_empty_list

    # Test 7: Different tokens produce different hash
    tokens1 = torch.tensor([1, 2, 3], device="cpu")
    tokens2 = torch.tensor([4, 5, 6], device="cpu")
    hash1 = db._hash_tokens(tokens1, prefix_hash=100)
    hash2 = db._hash_tokens(tokens2, prefix_hash=100)
    assert hash1 != hash2

    # Test 8: Invalid token type should raise ValueError
    with pytest.raises(ValueError, match="Unsupported tokens type"):
        db._hash_tokens("invalid", prefix_hash=100)

    # Test 9: Deterministicity with same inputs multiple times
    tokens = torch.tensor([10, 20, 30, 40], device="cpu")
    prefix = 999
    extra = ["a", "b", "c"]

    hashes = [db._hash_tokens(tokens, prefix, extra) for _ in range(10)]
    assert len(set(hashes)) == 1, "All hashes should be identical"
