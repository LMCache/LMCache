# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``skip_mm_storage`` option in ReqMeta.from_request_tracker.

When set, storage is limited to the contiguous text-only prefix before the
first multimodal token, so KV for varying mm content is not written to CPU.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    ReqMeta,
    RequestTracker,
)


@dataclass
class _FakePlaceholderRange:
    offset: int
    length: int


def _make_tracker(prompt_len: int, mm_positions=None) -> RequestTracker:
    return RequestTracker(
        req_id="test-req",
        prompt_len=prompt_len,
        token_ids=list(range(prompt_len)),
        allocated_block_ids=list(range(prompt_len // 16 + 1)),
        num_saved_tokens=0,
        mm_positions=mm_positions,
    )


def test_no_mm_positions_is_noop():
    tracker = _make_tracker(prompt_len=512)
    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert len(meta.token_ids) == 512


def test_skip_mm_storage_truncates_to_first_mm_offset():
    # System prompt: 0..127, image: 128..15999, query: 16000..16399
    mm_positions = [_FakePlaceholderRange(offset=128, length=15872)]
    tracker = _make_tracker(prompt_len=16400, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert len(meta.token_ids) == 128


def test_skip_mm_storage_off_stores_everything():
    mm_positions = [_FakePlaceholderRange(offset=128, length=15872)]
    tracker = _make_tracker(prompt_len=16400, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=False
    )
    assert meta is not None
    # Without the flag, we store as much as chunk-aligned full input
    assert len(meta.token_ids) == 16400 // 64 * 64


def test_skip_mm_storage_aligns_to_chunk_size():
    # First mm token at 130, chunk_size=64 -> aligned floor = 128
    mm_positions = [_FakePlaceholderRange(offset=130, length=1000)]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert len(meta.token_ids) == 128


def test_skip_mm_storage_picks_first_of_multiple_mm_ranges():
    # Two images: at 200 and at 800; we should clamp to 200 (chunk-aligned to 192)
    mm_positions = [
        _FakePlaceholderRange(offset=800, length=400),
        _FakePlaceholderRange(offset=200, length=400),
    ]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    # min offset = 200, aligned floor to 64 -> 192
    assert len(meta.token_ids) == 192


def test_skip_mm_storage_zero_offset_skips_save():
    # mm token starts at position 0 -> nothing to save
    mm_positions = [_FakePlaceholderRange(offset=0, length=2000)]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    # token_ids should be empty since no text prefix before mm
    if meta is not None:
        assert len(meta.token_ids) == 0
