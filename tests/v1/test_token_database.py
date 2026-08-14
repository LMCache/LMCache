# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import patch
import hashlib
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


class StubTokenizer:
    """Tokenizer stand-in that encodes any text to a fixed list of token ids.

    ``SegmentTokenDatabase`` only ever encodes ``blend_special_str``, so a
    fixed encoding is enough to pin down the separator -- and therefore its
    length -- without downloading a real tokenizer.
    """

    def __init__(self, encoded: list[int]) -> None:
        self.encoded = encoded

    def encode(self, text: str) -> list[int]:
        return list(self.encoded)


def make_segment_token_database(encoded_sep: list[int]) -> SegmentTokenDatabase:
    """Build a ``SegmentTokenDatabase`` with a deterministic separator.

    Args:
        encoded_sep: What the tokenizer returns for ``blend_special_str``.
            The database drops the leading token, so the separator ends up
            being ``encoded_sep[1:]``.

    Returns:
        A database backed by a stub tokenizer, so neither network access nor
        Hugging Face credentials are required.
    """
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata()
    with patch("lmcache.v1.token_database.AutoTokenizer") as auto_tokenizer:
        auto_tokenizer.from_pretrained.return_value = StubTokenizer(encoded_sep)
        return SegmentTokenDatabase(cfg, metadata)


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
    sys_hash = db._hash_tokens(sys_tokens)
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_hash = db._hash_tokens(token_chunk)
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_hash = db._hash_tokens(query_tokens)
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


def test_segment_token_database_splits_on_separator() -> None:
    """Separators split the input, and their tokens belong to no chunk."""
    db = make_segment_token_database([0, 100, 200])
    assert len(db.sep_tokens) == 2

    left = torch.tensor([1, 2, 3], dtype=torch.long)
    right = torch.tensor([4, 5], dtype=torch.long)
    tokens = torch.cat([left, db.sep_tokens, right])

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert [(start, end) for start, end, _ in results] == [(0, 3), (5, 7)]


def test_segment_token_database_input_shorter_than_separator() -> None:
    """Input too short to hold a separator comes back as one whole chunk.

    Regression test: the short-input guard used to fall through into the
    sliding-window match, where ``torch.Tensor.unfold`` raises a RuntimeError
    because it cannot build a window wider than the input.
    """
    db = make_segment_token_database([0, 100, 200, 300])
    assert len(db.sep_tokens) == 3

    tokens = torch.tensor([7, 8], dtype=torch.long)

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert [(start, end) for start, end, _ in results] == [(0, 2)]


def test_segment_token_database_empty_separator() -> None:
    """An empty separator leaves the input unsplit.

    ``sep_tokens`` is empty whenever the tokenizer encodes
    ``blend_special_str`` to a single token, because the database drops the
    leading one. Regression test: the guard for that case used to fall through
    into the sliding-window match, where a zero-width window matches at every
    position and shreds the input into single-token and empty chunks.
    """
    db = make_segment_token_database([0])
    assert len(db.sep_tokens) == 0

    tokens = generate_tokens(8, "cpu")

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert [(start, end) for start, end, _ in results] == [(0, 8)]


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
