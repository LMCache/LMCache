# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.kv_cache_groups import LMCKVCacheGroup, LMCKVCacheGroups


def test_lmc_kv_cache_groups_default_to_one_engine_group():
    assert LMCKVCacheGroups().num_engine_kv_cache_groups == 1
    assert LMCKVCacheGroups().per_layer_engine_group_indices(1) is None


def test_lmc_kv_cache_groups_build_per_layer_engine_group_indices():
    groups = LMCKVCacheGroups.from_groups(
        [
            LMCKVCacheGroup(0, ("layer.0", "layer.2"), (0, 2)),
            LMCKVCacheGroup(1, ("layer.1", "layer.3"), (1, 3)),
        ]
    )

    assert groups.num_engine_kv_cache_groups == 2
    assert groups.per_layer_engine_group_indices(4) == [0, 1, 0, 1]


def test_lmc_kv_cache_groups_serialize_round_trip():
    groups = LMCKVCacheGroups.from_groups(
        [
            LMCKVCacheGroup(0, ("layer.0", "layer.2"), (0, 2)),
            LMCKVCacheGroup(1, ("layer.1", "layer.3"), (1, 3)),
        ]
    )

    decoded = LMCKVCacheGroups.deserialize(groups.serialize())

    assert decoded == groups


def test_lmc_kv_cache_groups_reject_missing_layers():
    groups = LMCKVCacheGroups.from_groups(
        [
            LMCKVCacheGroup(0, ("layer.0",), (0,)),
            LMCKVCacheGroup(1, ("layer.1",), (1,)),
        ]
    )

    try:
        groups.per_layer_engine_group_indices(3)
    except ValueError as exc:
        assert "did not cover" in str(exc)
    else:
        raise AssertionError("Expected missing layer validation to fail")
