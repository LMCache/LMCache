# SPDX-License-Identifier: Apache-2.0
# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.group_view import (
    LMCacheGroupView,
    expand_block_ids_to_lmc_groups,
    get_engine_group_indices,
    num_engine_groups,
    num_lmc_kv_cache_groups,
)


def test_lmc_kv_cache_groups_default_to_one_engine_group():
    assert num_engine_groups([]) == 1
    assert num_lmc_kv_cache_groups([]) == 1
    assert get_engine_group_indices([], 1) is None


def test_lmc_kv_cache_groups_build_per_layer_engine_group_indices():
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(1, (1, 3)),
    ]

    assert num_engine_groups(groups) == 2
    assert num_lmc_kv_cache_groups(groups) == 2
    assert get_engine_group_indices(groups, 4) == [0, 1, 0, 1]


def test_lmc_kv_cache_groups_expand_block_ids_to_lmc_groups():
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(0, (4,)),
        LMCacheGroupView(1, (1, 3)),
    ]

    assert expand_block_ids_to_lmc_groups(groups, [[10, 11], [20, 21]]) == [
        [10, 11],
        [10, 11],
        [20, 21],
    ]


def test_lmc_kv_cache_groups_msgspec_round_trip():
    """The groups encode/decode losslessly via msgspec (the IPC path)."""
    groups = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(1, (1, 3)),
    ]

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(groups), type=list[LMCacheGroupView]
    )

    assert decoded == groups


def test_lmc_kv_cache_groups_reject_missing_layers():
    groups = [
        LMCacheGroupView(0, (0,)),
        LMCacheGroupView(1, (1,)),
    ]

    try:
        get_engine_group_indices(groups, 3)
    except ValueError as exc:
        assert "did not cover" in str(exc)
    else:
        raise AssertionError("Expected missing layer validation to fail")
