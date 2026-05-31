# SPDX-License-Identifier: Apache-2.0
# Third Party
import msgspec

# First Party
from lmcache.v1.kv_cache_groups import LMCacheKVGroup, LMCacheKVSpec


def test_lmc_kv_cache_groups_default_to_one_engine_group():
    assert LMCacheKVSpec().num_hybrid_block_groups == 1
    assert LMCacheKVSpec().num_lmc_kv_cache_groups == 1
    assert LMCacheKVSpec().get_per_layer_hybrid_block_group_indices(1) is None
    assert LMCacheKVSpec().hybrid_block_group_ids_by_lmc_group() == (0,)


def test_lmc_kv_cache_groups_build_per_layer_engine_group_indices():
    groups = LMCacheKVSpec.from_groups(
        [
            LMCacheKVGroup(0, (0, 2)),
            LMCacheKVGroup(1, (1, 3)),
        ]
    )

    assert groups.num_hybrid_block_groups == 2
    assert groups.num_lmc_kv_cache_groups == 2
    assert groups.get_per_layer_hybrid_block_group_indices(4) == [0, 1, 0, 1]
    assert groups.hybrid_block_group_ids_by_lmc_group() == (0, 1)


def test_lmc_kv_cache_groups_expand_block_ids_to_lmc_groups():
    groups = LMCacheKVSpec.from_groups(
        [
            LMCacheKVGroup(0, (0, 2)),
            LMCacheKVGroup(0, (4,)),
            LMCacheKVGroup(1, (1, 3)),
        ]
    )

    assert groups.expand_block_ids_to_lmc_groups([[10, 11], [20, 21]]) == [
        [10, 11],
        [10, 11],
        [20, 21],
    ]


def test_lmc_kv_cache_groups_msgspec_round_trip():
    """The spec encodes/decodes losslessly via msgspec (the IPC path)."""
    groups = LMCacheKVSpec.from_groups(
        [
            LMCacheKVGroup(0, (0, 2)),
            LMCacheKVGroup(1, (1, 3)),
        ]
    )

    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(groups), type=LMCacheKVSpec)

    assert decoded == groups


def test_lmc_kv_cache_groups_reject_missing_layers():
    groups = LMCacheKVSpec.from_groups(
        [
            LMCacheKVGroup(0, (0,)),
            LMCacheKVGroup(1, (1,)),
        ]
    )

    try:
        groups.get_per_layer_hybrid_block_group_indices(3)
    except ValueError as exc:
        assert "did not cover" in str(exc)
    else:
        raise AssertionError("Expected missing layer validation to fail")
