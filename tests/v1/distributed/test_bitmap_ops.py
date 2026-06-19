# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fold / unfold prefix-cache hit logic."""

# Standard
import random

# Third Party
import pytest

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import TrimPolicy
from lmcache.v1.distributed.bitmap_ops import (
    FULL_ATTENTION_WINDOW,
    fold_unfold,
    fold_unfold_ranked,
    merge_bitmaps,
    select_retained,
    unfold_range,
)
from lmcache.v1.distributed.bitmap_ops.fold import (
    _fold_unfold_ranked_python,
    _fold_unfold_ranked_torch,
)


def _make_presence(num_chunks: int, present_per_group: list[list[int]]) -> Bitmap:
    """Build a group-major presence bitmap.

    Args:
        num_chunks: chunks per group.
        present_per_group: present_per_group[g] is the list of chunk indices
            available for object group g.

    Returns:
        A group-major Bitmap of length ``len(present_per_group) * num_chunks``.
    """
    bm = Bitmap(len(present_per_group) * num_chunks)
    for group_idx, chunks in enumerate(present_per_group):
        base = group_idx * num_chunks
        for j in chunks:
            bm.set(base + j)
    return bm


# --------------------------------------------------------------------------- #
# unfold_range                                                                 #
# --------------------------------------------------------------------------- #


def test_unfold_full_attention_needs_whole_prefix():
    assert unfold_range(4, FULL_ATTENTION_WINDOW) == (0, 4)
    assert unfold_range(4, 0) == (0, 4)


def test_unfold_window_needs_only_last_w():
    assert unfold_range(4, 2) == (2, 4)
    assert unfold_range(1, 2) == (0, 1)  # window larger than prefix
    assert unfold_range(5, 1) == (4, 5)  # mamba: last chunk only


def test_unfold_empty_prefix():
    assert unfold_range(0, FULL_ATTENTION_WINDOW) == (0, 0)
    assert unfold_range(0, 2) == (0, 0)


# --------------------------------------------------------------------------- #
# fold_unfold — single group reduces to leading-ones                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "present,expected_hit",
    [
        ([0, 1, 2], 3),  # full contiguous prefix
        ([0, 1, 3], 2),  # gap at 2 caps the prefix
        ([], 0),  # nothing present
        ([1, 2], 0),  # missing chunk 0 -> empty prefix
    ],
)
def test_single_full_group_equals_leading_ones(present, expected_hit):
    num_chunks = 4
    found = _make_presence(num_chunks, [present])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW])
    assert hit == expected_hit
    # equals the plain PREFIX leading-ones count on the same bitmap
    assert hit == found.count_leading_ones()
    # retained mask is exactly the first `hit` chunks
    assert mask.get_indices_list() == list(range(expected_hit))


# --------------------------------------------------------------------------- #
# fold_unfold — worked full + sliding-window example                           #
# --------------------------------------------------------------------------- #


def test_full_plus_sliding_window_worked_example():
    # N=5; group A full present {0,1,2,3}; group B sliding-window w=2 {2,3,4}.
    # A blocks length 5 (chunk 4 missing); B's last-2 window at L=4 is {2,3} (present).
    num_chunks = 5
    found = _make_presence(num_chunks, [[0, 1, 2, 3], [2, 3, 4]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 4
    # A (full) needs chunks 0..3 -> flat 0,1,2,3 ; B (w=2) needs 2..3 -> flat 7,8
    assert mask.get_indices_list() == [0, 1, 2, 3, 7, 8]


def test_sliding_window_does_not_block_long_prefix_when_tail_present():
    # SW group missing early chunks but holding the tail still serves a long hit.
    num_chunks = 6
    found = _make_presence(num_chunks, [[0, 1, 2, 3, 4, 5], [4, 5]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 6
    # full needs 0..5 ; window-2 needs 4..5 -> flat 6*1 + {4,5} = {10,11}
    assert mask.get_indices_list() == [0, 1, 2, 3, 4, 5, 10, 11]


def test_mamba_window_one():
    # mamba == window 1: only the last chunk of the prefix is needed.
    num_chunks = 4
    found = _make_presence(num_chunks, [[0, 1, 2, 3], [3]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 1])
    assert hit == 4
    assert mask.get_indices_list() == [0, 1, 2, 3, 7]  # full 0..3 + mamba {3}


# --------------------------------------------------------------------------- #
# fold_unfold — all-full reduces to require-all intersection                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "group_a,group_b,expected_hit",
    [
        ([0, 1, 2, 3], [0, 1, 2, 3], 4),  # both full -> full
        ([0, 1, 2, 3], [0, 1], 2),  # B caps at 2
        ([0, 1], [0, 1, 2, 3], 2),  # A caps at 2
        ([0, 2, 3], [0, 1, 2, 3], 1),  # A gap at 1 caps at 1
    ],
)
def test_all_full_is_require_all_intersection(group_a, group_b, expected_hit):
    num_chunks = 4
    found = _make_presence(num_chunks, [group_a, group_b])
    windows = [FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW]
    hit, mask = fold_unfold(found, num_chunks, windows)
    assert hit == expected_hit
    # both groups retain the same first `hit` chunks
    expected = list(range(expected_hit)) + [num_chunks + j for j in range(expected_hit)]
    assert mask.get_indices_list() == expected


# --------------------------------------------------------------------------- #
# fold_unfold — edges                                                          #
# --------------------------------------------------------------------------- #


def test_zero_chunks():
    found = Bitmap(0)
    hit, mask = fold_unfold(found, 0, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 0
    assert mask.get_indices_list() == []


# --------------------------------------------------------------------------- #
# fold_unfold_ranked — group x chunk x kv_rank layout                          #
# --------------------------------------------------------------------------- #


def _make_ranked(
    num_chunks: int,
    num_ranks: int,
    present_per_group: list[list[tuple[int, int]]],
) -> Bitmap:
    """Build a group-major / chunk-major / rank-minor presence bitmap.

    present_per_group[g] is the list of ``(chunk, rank)`` present for group g.
    """
    num_groups = len(present_per_group)
    stride = num_chunks * num_ranks
    bm = Bitmap(num_groups * stride)
    for group_idx, cells in enumerate(present_per_group):
        gbase = group_idx * stride
        for chunk, rank in cells:
            bm.set(gbase + chunk * num_ranks + rank)
    return bm


def test_ranked_chunk_present_only_if_all_ranks_present():
    # 1 full group, 2 ranks, 3 chunks. chunk1 is missing rank 1 -> not present.
    present = [[(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]]
    found = _make_ranked(3, 2, present)
    hit, mask = fold_unfold_ranked(found, 3, 2, [FULL_ATTENTION_WINDOW])
    assert hit == 1  # only chunk 0 has both ranks; chunk1 gap caps the prefix
    assert mask.get_indices_list() == [0, 1]  # both ranks of chunk 0


def test_ranked_reduces_to_unranked_when_one_rank():
    # num_ranks == 1 must match fold_unfold exactly.
    found_unranked = _make_presence(5, [[0, 1, 2, 3], [2, 3, 4]])
    found_ranked = _make_ranked(
        5, 1, [[(c, 0) for c in [0, 1, 2, 3]], [(c, 0) for c in [2, 3, 4]]]
    )
    hit_u, mask_u = fold_unfold(found_unranked, 5, [FULL_ATTENTION_WINDOW, 2])
    hit_r, mask_r = fold_unfold_ranked(found_ranked, 5, 1, [FULL_ATTENTION_WINDOW, 2])
    assert hit_u == hit_r == 4
    assert mask_u.get_indices_list() == mask_r.get_indices_list()


def test_ranked_full_plus_sw_expands_all_ranks():
    # 2 groups, 2 ranks, 4 chunks. group0 full all present; group1 SW w=1 all present.
    g0 = [(c, r) for c in range(4) for r in range(2)]
    g1 = [(c, r) for c in range(4) for r in range(2)]
    found = _make_ranked(4, 2, [g0, g1])
    hit, mask = fold_unfold_ranked(found, 4, 2, [FULL_ATTENTION_WINDOW, 1])
    assert hit == 4
    # group0 full -> chunks 0..3 (ranks 0,1): flat 0..7
    # group1 w=1 -> chunk 3 only (ranks 0,1): group base = 4*2 = 8, chunk3 -> 8+6,8+7
    assert mask.get_indices_list() == [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]


def test_ranked_invalid_num_ranks_raises():
    with pytest.raises(ValueError):
        fold_unfold_ranked(Bitmap(0), 0, 0, [FULL_ATTENTION_WINDOW])


def test_empty_group_windows_raises():
    with pytest.raises(ValueError):
        fold_unfold(Bitmap(0), 0, [])


def test_negative_num_chunks_raises():
    with pytest.raises(ValueError):
        fold_unfold(Bitmap(0), -1, [FULL_ATTENTION_WINDOW])


def _bm(num_keys: int, set_indices: list[int]) -> Bitmap:
    bm = Bitmap(num_keys)
    for i in set_indices:
        bm.set(i)
    return bm


class TestSelectRetained:
    """select_retained picks the retained subset per policy: PREFIX trims at the
    first gap; any other policy keeps every set bit (gaps and all)."""

    def test_prefix_trims_at_first_gap(self):
        found = _bm(5, [0, 1, 3, 4])  # gap at index 2
        assert select_retained(found, 5, TrimPolicy.PREFIX).get_indices_list() == [0, 1]

    def test_sparse_keeps_all_found(self):
        found = _bm(5, [0, 2, 4])
        result = select_retained(found, 5, TrimPolicy.SPARSE).get_indices_list()
        assert result == [0, 2, 4]

    def test_segmented_prefix_keeps_all_found(self):
        found = _bm(5, [0, 1, 3, 4])  # gap at index 2
        result = select_retained(
            found, 5, TrimPolicy.SEGMENTED_PREFIX
        ).get_indices_list()
        assert result == [0, 1, 3, 4]


class TestMergeBitmaps:
    """merge_bitmaps always returns a num_keys-sized bitmap."""

    def test_empty_input_returns_sized_bitmap(self):
        """Empty input -> num_keys-sized all-zeros bitmap (not Bitmap(0)), so a
        downstream ``&`` with a same-sized mask never hits a size mismatch."""
        merged = merge_bitmaps([], 5)
        assert merged.popcount() == 0
        mask = Bitmap(5)
        mask.set(2)
        assert (merged & mask).popcount() == 0  # would raise on size mismatch

    def test_empty_generator_returns_sized_bitmap(self):
        """A generator is truthy even when empty; the result is still size-5."""
        merged = merge_bitmaps((b for b in []), 5)
        assert merged.popcount() == 0
        assert (merged & Bitmap(5)).popcount() == 0

    def test_union_of_bitmaps(self):
        """Non-empty inputs are OR-merged into one num_keys-sized bitmap."""
        a, b = Bitmap(5), Bitmap(5)
        a.set(0)
        b.set(3)
        assert merge_bitmaps([a, b], 5).get_indices_list() == [0, 3]


class TestRankedTorchMatchesPython:
    """The vectorized torch fold must match the pure-Python reference exactly."""

    @staticmethod
    def _result(fn, num_chunks, num_ranks, gw, present):
        nk = len(gw) * num_chunks * num_ranks
        bm = Bitmap(nk)
        bm.batched_set([i for i, p in enumerate(present) if p])
        hit, mask = fn(bm, num_chunks, num_ranks, gw)
        return hit, mask.get_indices_list()

    def test_random_equivalence(self):
        rng = random.Random(1234)
        for _ in range(300):
            num_groups = rng.randint(1, 4)
            num_chunks = rng.randint(1, 16)
            num_ranks = rng.randint(1, 3)
            gw = [rng.choice([-1, 1, 2, 3]) for _ in range(num_groups)]
            nk = num_groups * num_chunks * num_ranks
            present = [rng.random() < 0.6 for _ in range(nk)]
            hit_py, mask_py = self._result(
                _fold_unfold_ranked_python, num_chunks, num_ranks, gw, present
            )
            hit_t, mask_t = self._result(
                _fold_unfold_ranked_torch, num_chunks, num_ranks, gw, present
            )
            assert (hit_py, mask_py) == (hit_t, mask_t), (
                f"mismatch gw={gw} C={num_chunks} R={num_ranks}"
            )

    def test_dispatch_large_input_matches_reference(self):
        # Above the dispatch threshold the public API uses torch; it must still
        # match the Python reference (full + sliding-window groups, gappy data).
        gw = [-1, -1, 4, 4, 8, 1, -1, 2]  # 8 groups
        num_chunks, num_ranks = 64, 8  # 8 * 64 * 8 = 4096 keys >= threshold
        nk = len(gw) * num_chunks * num_ranks
        rng = random.Random(7)
        present = [rng.random() < 0.7 for _ in range(nk)]
        bm = Bitmap(nk)
        bm.batched_set([i for i, p in enumerate(present) if p])

        hit_pub, mask_pub = fold_unfold_ranked(bm, num_chunks, num_ranks, gw)
        hit_ref, mask_ref = self._result(
            _fold_unfold_ranked_python, num_chunks, num_ranks, gw, present
        )
        assert hit_pub == hit_ref
        assert mask_pub.get_indices_list() == mask_ref
