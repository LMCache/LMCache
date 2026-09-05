# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import token_database
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


class _StubTokenizer:
    """Minimal tokenizer stand-in for SegmentTokenDatabase.

    SegmentTokenDatabase derives its separator as
    ``tokenizer.encode(blend_special_str)[1:]`` -- the leading element is
    treated as a special start token and dropped. This stub returns a fixed
    leading id followed by the requested separator ids so that ``sep_tokens``
    is fully controlled without downloading a real tokenizer (the only existing
    SegmentTokenDatabase test is skipped unless Hugging Face credentials are
    available).
    """

    def __init__(self, sep_ids: list[int]) -> None:
        self._encoded = [0, *sep_ids]

    def encode(self, text: str) -> list[int]:
        return list(self._encoded)


def _make_segment_db(monkeypatch, sep_ids: list[int]) -> SegmentTokenDatabase:
    monkeypatch.setattr(
        token_database.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _StubTokenizer(sep_ids),
    )
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str="<sep>")
    metadata = dumb_metadata_with_model_name("test_model")
    return SegmentTokenDatabase(cfg, metadata)


def test_segment_process_tokens_shorter_than_separator(monkeypatch) -> None:
    """A sequence shorter than the separator must be returned as a single
    segment.

    Regression test: the splitting guard previously fell through to
    ``tensor.unfold``, which raises ``RuntimeError`` when the window is wider
    than the tensor, crashing on any request shorter than the separator.
    """
    db = _make_segment_db(monkeypatch, sep_ids=[101, 102])
    tokens = torch.tensor([7], dtype=torch.long)

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert len(results) == 1
    start, end, _ = results[0]
    assert (start, end) == (0, 1)


def test_segment_process_tokens_empty_separator(monkeypatch) -> None:
    """An empty separator must yield the whole sequence as a single segment.

    Regression test: with ``sep_len == 0`` the guard previously fell through
    and every sliding window matched vacuously, exploding the input into
    spurious empty/single-token segments.
    """
    db = _make_segment_db(monkeypatch, sep_ids=[])
    tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert len(results) == 1
    start, end, _ = results[0]
    assert (start, end) == (0, 4)


def test_segment_process_tokens_splits_on_separator(monkeypatch) -> None:
    """The normal multi-segment split path is unchanged by the guard fix."""
    db = _make_segment_db(monkeypatch, sep_ids=[101, 102])
    sep = torch.tensor([101, 102], dtype=torch.long)
    first = torch.tensor([1, 2, 3], dtype=torch.long)
    second = torch.tensor([4, 5], dtype=torch.long)
    tokens = torch.cat([first, sep, second])

    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert len(results) == 2
    # First segment covers the tokens before the separator.
    assert results[0][0] == 0
    assert results[0][1] == len(first)
    # Second segment starts after the dropped separator and runs to the end.
    assert results[1][0] == len(first) + len(sep)
    assert results[1][1] == len(tokens)
