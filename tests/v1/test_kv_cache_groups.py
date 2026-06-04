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
