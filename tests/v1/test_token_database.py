# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import convert_token_range_to_list
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


def test_process_tokens_returns_int_keys_for_bytes_hash_func() -> None:
    """process_tokens must produce int chunk_hash keys even when the underlying
    hash function returns bytes (e.g. sha256_cbor). This ensures downstream
    code such as msgpack serialisation always receives plain ints.
    """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu")
    metadata = dumb_metadata()
    db = ChunkedTokenDatabase(cfg, metadata)

    # Stub that returns bytes, mimicking sha256_cbor without requiring cbor2.
    db.hash_func = lambda x: hashlib.sha256(str(x).encode()).digest()

    tokens = generate_tokens(32, "cpu")
    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert len(results) > 0
    for _, _, hash_val in results:
        assert isinstance(hash_val, int), f"Expected int, got {type(hash_val)}"
        # Must fit in uint64 (msgpack range: 0 to 2**64 - 1)
        assert 0 <= hash_val <= 2**64 - 1


@pytest.mark.parametrize("chunk_size", [4, 16])
def test_kv_event_token_ids_cover_exactly_one_chunk(chunk_size) -> None:
    """The KV-event token slice must match the chunk it describes.

    ``process_tokens`` yields half-open ``[start, end)`` bounds while
    ``convert_tokens_to_list`` is inclusive of its end index, so passing ``end``
    straight through appends the first token of the *next* chunk to every
    non-final event and breaks ``len(token_ids) == block_size``.
    """
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_size, backend="cpu", save_unfull_chunk=True
    )
    db = ChunkedTokenDatabase(cfg, dumb_metadata())

    tokens = list(range(chunk_size * 2 + chunk_size // 2))
    results = list(db.process_tokens(tokens=tokens))
    assert len(results) > 1  # need at least one non-final chunk

    for start, end, _ in results:
        token_ids = convert_token_range_to_list(tokens, start, end)
        assert token_ids == tokens[start:end]
        assert len(token_ids) == end - start


def test_kv_event_chunk_hash_matches_supplied_hashes() -> None:
    """In the hashes variant the event carries this chunk's hash.

    ``hashes`` holds one entry per chunk while ``start``/``end`` are token
    offsets, so slicing ``hashes`` with them selects the wrong entries (every
    hash for the first chunk, none for later ones).
    """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
    db = ChunkedTokenDatabase(cfg, dumb_metadata())

    hashes = [1111, 2222, 3333]
    offsets = [256, 256, 256]
    results = list(db.process_tokens(hashes=hashes, offsets=offsets))

    assert len(results) == len(hashes)
    for (_, _, key), expected in zip(results, hashes, strict=False):
        assert [key.chunk_hash] == [expected]
