# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# First Party
from lmcache.integration.vllm.kv_cache_groups import lmcache_kv_cache_groups_from_vllm


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


def test_vllm_kv_cache_groups_conversion_defaults_to_one():
    groups = lmcache_kv_cache_groups_from_vllm(None)

    assert groups.groups == ()
    assert groups.num_engine_kv_cache_groups == 1


def test_vllm_kv_cache_groups_conversion_preserves_group_layers():
    groups = lmcache_kv_cache_groups_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0", "layer.2"]),
                MockKVCacheGroup(["layer.1", "layer.3"]),
            ]
        )
    )

    assert groups.num_engine_kv_cache_groups == 2
    assert groups.to_layout_hints(("layer.0", "layer.1", "layer.2", "layer.3")) == {
        "per_layer_engine_group_idx": [0, 1, 0, 1]
    }
