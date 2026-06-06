# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the connector-side per-group block-ID slicing + SWA-suffix
trimming + per-group APC-skip computation (scheduler-side, no server / GPU).

These cover the load-bearing arithmetic that moved out of the server in the
"hide vLLM info from LMCache" refactor: the connector now pre-trims SWA groups
to their trailing window and emits a per-group skip-block count, so the server
stays window- and logical-size-agnostic.
"""

# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


def _tracker(allocated_block_ids):
    """Minimal tracker stand-in: only ``allocated_block_ids`` is read by the
    geometry / slicing static methods under test."""
    return SimpleNamespace(allocated_block_ids=dict(allocated_block_ids))


# ----------------------------------------------------------------------------
# _group_ratio / _group_geometry
# ----------------------------------------------------------------------------


def test_group_ratio_gcd_grain():
    # logical 256 over GCD block size 4 -> ratio 64
    assert LMCacheMPRequestMetadata._group_ratio(0, [256, 64], 4) == 64
    assert LMCacheMPRequestMetadata._group_ratio(1, [256, 64], 4) == 16


def test_group_ratio_defaults_to_one():
    # No per-group logical sizes (non-hybrid) -> ratio 1 everywhere.
    assert LMCacheMPRequestMetadata._group_ratio(0, [], 16) == 1
    # Out-of-range index -> 1.
    assert LMCacheMPRequestMetadata._group_ratio(5, [256], 4) == 1


def test_group_geometry_full_attention():
    # logical 256, GCD 4, blocks_in_chunk 256 (chunk 1024 tokens / 4):
    # ratio 64, full_bpc 4, no window -> suffix == full.
    ratio, full_bpc, suffix_bpc = LMCacheMPRequestMetadata._group_geometry(
        0, [256], [0], 4, 256
    )
    assert (ratio, full_bpc, suffix_bpc) == (64, 4, 4)


def test_group_geometry_swa_suffix():
    # logical 64 (ratio 16), full_bpc = 256/16 = 16, window 128 -> ceil(128/64)=2.
    ratio, full_bpc, suffix_bpc = LMCacheMPRequestMetadata._group_geometry(
        0, [64], [128], 4, 256
    )
    assert (ratio, full_bpc, suffix_bpc) == (16, 16, 2)


# ----------------------------------------------------------------------------
# _slice_block_ids: own-grain slice + SWA-suffix trimming
# ----------------------------------------------------------------------------


def test_slice_full_attention_no_trim():
    # Single group, ratio 1, full chunk. GCD range [0, 8) over 2 chunks of 4.
    tracker = _tracker({0: list(range(100, 120))})
    sliced = LMCacheMPRequestMetadata._slice_block_ids(
        tracker,
        num_engine_groups=1,
        start=0,
        end=8,
        logical_block_sizes=[],  # non-hybrid -> ratio 1
        sliding_windows=[],
        vllm_block_size=16,
        blocks_in_chunk=4,
    )
    # Whole [0,8) range, untrimmed.
    assert sliced == [list(range(100, 108))]


def test_slice_swa_keeps_trailing_window_per_chunk():
    # One SWA group: logical 64, GCD 4 -> ratio 16, full_bpc = 16 per chunk,
    # window 128 -> suffix 2. Two chunks. GCD range [0, 2*64) = [0,128).
    # In own grain: [0//16, 128//16) = [0, 8) -> 8 own-grain blocks = 2 chunks
    # of full_bpc 4? No: blocks_in_chunk is GCD-grain (64), full_bpc=64/16=4.
    # Use blocks_in_chunk=64 (chunk 1024 tokens / 16 GCD blocks... keep simple):
    # logical 64, vllm_bs 4 -> ratio 16; blocks_in_chunk 64 -> full_bpc 4,
    # window 64 -> ceil(64/64)=1 suffix.
    own = list(range(200, 240))  # 40 own-grain block ids available
    tracker = _tracker({0: own})
    sliced = LMCacheMPRequestMetadata._slice_block_ids(
        tracker,
        num_engine_groups=1,
        start=0,
        end=128,  # 2 chunks of blocks_in_chunk=64
        logical_block_sizes=[64],
        sliding_windows=[64],  # window == one logical block -> suffix 1
        vllm_block_size=4,
        blocks_in_chunk=64,
    )
    # own slice = own[0:8] (128//16=8), full_bpc=4, suffix=1:
    # chunk0 own[0:4] -> keep last 1 = own[3]; chunk1 own[4:8] -> own[7].
    assert sliced == [[own[3], own[7]]]


def test_slice_mixed_groups_trim_only_swa():
    # Group 0: full attention (logical 256, ratio 64, full_bpc=4, no window).
    # Group 1: SWA (logical 64, ratio 16, full_bpc=16, window 128 -> suffix 2).
    # One chunk: blocks_in_chunk = 256 GCD blocks, range [0, 256).
    g0 = list(range(1000, 1004))  # group0 own grain: 256//64 = 4 ids
    g1 = list(range(2000, 2016))  # group1 own grain: 256//16 = 16 ids
    tracker = _tracker({0: g0, 1: g1})
    sliced = LMCacheMPRequestMetadata._slice_block_ids(
        tracker,
        num_engine_groups=2,
        start=0,
        end=256,
        logical_block_sizes=[256, 64],
        sliding_windows=[0, 128],
        vllm_block_size=4,
        blocks_in_chunk=256,
    )
    # group0: full 4 ids untrimmed; group1: keep trailing 2 of 16.
    assert sliced[0] == g0
    assert sliced[1] == g1[14:16]


# ----------------------------------------------------------------------------
# GetRetrieveMetadata: per-group skip_blocks_per_group
# ----------------------------------------------------------------------------


def _retrieve_tracker(*, allocated, block_hashes_len, vllm_hit, lmcache_hit, tokens):
    return SimpleNamespace(
        request_id="r0",
        cache_salt="",
        all_token_ids=list(range(tokens)),
        block_hashes=list(range(block_hashes_len)),
        allocated_block_ids=dict(allocated),
        num_stored_blocks=0,
        num_vllm_hit_blocks=vllm_hit,
        num_lmcache_hit_blocks=lmcache_hit,
        is_ready_for_retrieving=lambda: True,
    )


def test_retrieve_skip_blocks_per_group_swa():
    # blocks_in_chunk = 256 (GCD). vllm hit 0, lmcache hit 256 -> retrieve 1 chunk.
    # Group 0 full (logical 256), group 1 SWA (logical 64, window 128).
    # With vllm_hit=0 -> apc overlap 0 -> skip all zero.
    tracker = _retrieve_tracker(
        allocated={0: list(range(4)), 1: list(range(16))},
        block_hashes_len=256,
        vllm_hit=0,
        lmcache_hit=256,
        tokens=256 * 4,
    )
    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        blocks_in_chunk=256,
        vllm_block_size=4,
        num_engine_groups=2,
        logical_block_sizes=[256, 64],
        sliding_windows=[0, 128],
    )
    assert meta is not None
    assert meta.op.skip_blocks_per_group == [0, 0]


def test_retrieve_skip_absorbed_by_swa_trim():
    # vllm_hit within the first chunk produces an APC overlap; the SWA group's
    # dropped leading blocks should absorb it (skip clamps to 0), while a full
    # group still carries the block skip.
    # blocks_in_chunk=256 GCD; start floors num_vllm_hit to chunk -> 0.
    # vllm_hit=64 (GCD blocks) within chunk 0, lmcache_hit=256.
    tracker = _retrieve_tracker(
        allocated={0: list(range(4)), 1: list(range(16))},
        block_hashes_len=256,
        vllm_hit=64,
        lmcache_hit=256,
        tokens=256 * 4,
    )
    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        blocks_in_chunk=256,
        vllm_block_size=4,
        num_engine_groups=2,
        logical_block_sizes=[256, 64],
        sliding_windows=[0, 128],
    )
    assert meta is not None
    # apc_overlap_blocks (GCD) = 64; tokens = 64*4 = 256.
    # group0 full: logical 256 -> skip 256//256 = 1 block, suffix_offset 0 -> 1.
    # group1 SWA: logical 64 -> skip 256//64 = 4 blocks; full_bpc 16, suffix 2
    #   -> suffix_offset 14; 4 - 14 < 0 -> clamped to 0.
    assert meta.op.skip_blocks_per_group == [1, 0]
