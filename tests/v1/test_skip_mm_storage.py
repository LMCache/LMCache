# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``skip_mm_storage`` option in ReqMeta.from_request_tracker.

When set, the save path is capped to the contiguous text-only prefix before
the first multimodal token while the load path remains unaffected.
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
    # Full chunk-aligned token set on the load path; no save cap needed.
    assert len(meta.token_ids) == 512
    assert meta.save_spec.max_save_tokens is None


def test_skip_mm_storage_caps_save_at_first_mm_offset():
    # System prompt: 0..127, image: 128..15999, query: 16000..16399
    mm_positions = [_FakePlaceholderRange(offset=128, length=15872)]
    tracker = _make_tracker(prompt_len=16400, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    # Load path keeps the full chunk-aligned sequence so retrieve()
    # can index up to lmcache_cached_tokens without going out of range.
    assert len(meta.token_ids) == 16400 // 64 * 64
    # Save path is capped at the first mm offset.
    assert meta.save_spec.max_save_tokens == 128


def test_skip_mm_storage_off_stores_everything():
    mm_positions = [_FakePlaceholderRange(offset=128, length=15872)]
    tracker = _make_tracker(prompt_len=16400, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=False
    )
    assert meta is not None
    assert len(meta.token_ids) == 16400 // 64 * 64
    assert meta.save_spec.max_save_tokens is None


def test_skip_mm_storage_aligns_to_chunk_size():
    # First mm token at 130, chunk_size=64 -> aligned floor = 128
    mm_positions = [_FakePlaceholderRange(offset=130, length=1000)]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert meta.save_spec.max_save_tokens == 128


def test_skip_mm_storage_picks_first_of_multiple_mm_ranges():
    # Two images: at 200 and at 800; we cap to 200 (chunk-aligned to 192)
    mm_positions = [
        _FakePlaceholderRange(offset=800, length=400),
        _FakePlaceholderRange(offset=200, length=400),
    ]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert meta.save_spec.max_save_tokens == 192


def test_skip_mm_storage_zero_offset_caps_to_zero():
    # mm token starts at position 0 -> save cap is 0 (nothing stored).
    mm_positions = [_FakePlaceholderRange(offset=0, length=2000)]
    tracker = _make_tracker(prompt_len=2000, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    assert meta.save_spec.max_save_tokens == 0


def test_load_path_unaffected_by_skip_mm_storage():
    """Regression: token_ids must remain full-length so load can index into it."""
    mm_positions = [_FakePlaceholderRange(offset=128, length=15872)]
    tracker = _make_tracker(prompt_len=16400, mm_positions=mm_positions)

    meta = ReqMeta.from_request_tracker(
        tracker, block_size=16, lmcache_chunk_size=64, skip_mm_storage=True
    )
    assert meta is not None
    # If a prior lookup found 16320 cached tokens, tokens[:16320] must
    # not silently shrink below that.
    assert len(meta.token_ids) >= 16320
