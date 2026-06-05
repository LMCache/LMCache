# SPDX-License-Identifier: Apache-2.0
# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.group_view import (
    LMCacheGroupView,
    expand_block_ids_to_views,
    get_engine_group_indices,
    num_engine_groups,
    num_group_views,
    slice_block_ids_per_group,
)


def test_group_views_default_to_one_engine_group():
    assert num_engine_groups([]) == 1
    assert num_group_views([]) == 1
    assert get_engine_group_indices([], 1) is None


def test_group_views_build_per_layer_engine_group_indices():
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(1, (1, 3)),
    ]

    assert num_engine_groups(groups) == 2
    assert num_group_views(groups) == 2
    assert get_engine_group_indices(groups, 4) == [0, 1, 0, 1]


def test_group_views_expand_block_ids_to_views():
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(0, (4,)),
        LMCacheGroupView(1, (1, 3)),
    ]

    assert expand_block_ids_to_views(groups, [[10, 11], [20, 21]]) == [
        [10, 11],
        [10, 11],
        [20, 21],
    ]


def test_group_views_msgspec_round_trip():
    """The groups encode/decode losslessly via msgspec (the IPC path)."""
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(1, (1, 3)),
    ]

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(groups), type=list[LMCacheGroupView]
    )

    assert decoded == groups


def test_group_views_exclude_uncovered_layers():
    """Layers not referenced by any group are tagged EXCLUDED_ENGINE_GROUP.

    Cross-layer KV-sharing layers (e.g. google/gemma-4-E4B-it) alias a target
    owner's KV cache and are intentionally left out of every group; downstream
    grouping skips them rather than treating partial coverage as an error.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    groups = [
        LMCacheGroupView(0, (0,)),
        LMCacheGroupView(1, (1,)),
    ]

    # Layer 2 is not covered by any group -> excluded, not an error.
    assert get_engine_group_indices(groups, 3) == [0, 1, EXCLUDED_ENGINE_GROUP]


def test_group_views_reject_out_of_range_layer():
    groups = [LMCacheGroupView(0, (0, 5))]

    try:
        get_engine_group_indices(groups, 3)
    except ValueError as exc:
        assert "outside registered layer range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range layer index to fail")


def test_slice_block_ids_uniform_block_sizes():
    """Groups sharing the base block size slice to equal counts."""
    allocated = {0: list(range(16)), 1: list(range(100, 116))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_block_sizes=[16, 16],
        base_block_size=16,
        start_block_idx=0,
        end_block_idx=16,
    )
    assert sliced == [list(range(16)), list(range(100, 116))]


def test_slice_block_ids_heterogeneous_block_sizes():
    """A block_size-32 group gets half the IDs of a block_size-16 group.

    The range [0, 16) spans 256 tokens: the block_size-16 group needs
    16 block IDs, the block_size-32 group 8, for the same token span.
    """
    allocated = {0: list(range(16)), 1: list(range(8))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_block_sizes=[16, 32],
        base_block_size=16,
        start_block_idx=0,
        end_block_idx=16,
    )
    assert sliced == [list(range(16)), list(range(8))]


def test_slice_block_ids_nonzero_start_offset():
    """Start/end offsets are divided per group by the block factor."""
    allocated = {0: list(range(32)), 1: list(range(16))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_block_sizes=[16, 32],
        base_block_size=16,
        start_block_idx=16,
        end_block_idx=32,
    )
    assert sliced == [list(range(16, 32)), list(range(8, 16))]


def test_slice_block_ids_missing_group_yields_empty():
    """A group with no allocated block IDs slices to an empty list."""
    allocated = {0: list(range(16))}  # group 1 absent
    sliced = slice_block_ids_per_group(
        allocated,
        group_block_sizes=[16, 16],
        base_block_size=16,
        start_block_idx=0,
        end_block_idx=16,
    )
    assert sliced == [list(range(16)), []]


def test_slice_block_ids_misaligned_range_raises():
    """A range that is not a whole number of a group's blocks is rejected."""
    allocated = {0: list(range(8)), 1: list(range(8))}
    # group 1 block_size 48 -> factor 3; end=8 is not a multiple of 3.
    try:
        slice_block_ids_per_group(
            allocated,
            group_block_sizes=[16, 48],
            base_block_size=16,
            start_block_idx=0,
            end_block_idx=8,
        )
    except ValueError as exc:
        assert "does not align" in str(exc)
    else:
        raise AssertionError("Expected misaligned range to fail")
