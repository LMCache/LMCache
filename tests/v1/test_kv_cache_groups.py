# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.kv_cache_groups import LMCKVCacheGroup, LMCKVCacheGroups


def test_lmc_kv_cache_groups_default_to_one_engine_group():
    assert LMCKVCacheGroups().num_engine_kv_cache_groups == 1
    assert LMCKVCacheGroups().to_layout_hints(("layer.0",)) is None


def test_lmc_kv_cache_groups_build_layout_hints():
    groups = LMCKVCacheGroups.from_groups(
        [
            LMCKVCacheGroup(0, ("layer.0", "layer.2")),
            LMCKVCacheGroup(1, ("layer.1", "layer.3")),
        ]
    )

    assert groups.num_engine_kv_cache_groups == 2
    assert groups.to_layout_hints(("layer.0", "layer.1", "layer.2", "layer.3")) == {
        "per_layer_engine_group_idx": [0, 1, 0, 1]
    }


def test_lmc_kv_cache_groups_reject_missing_layers():
    groups = LMCKVCacheGroups.from_groups(
        [
            LMCKVCacheGroup(0, ("layer.0",)),
            LMCKVCacheGroup(1, ("layer.1",)),
        ]
    )

    try:
        groups.to_layout_hints(("layer.0", "layer.2"))
    except ValueError as exc:
        assert "did not cover" in str(exc)
    else:
        raise AssertionError("Expected missing layer validation to fail")
