# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    inflated_lmcache_kv_cache_groups_from_vllm,
    lmcache_kv_cache_groups_from_vllm,
)


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
        ),
        registered_layer_names=("layer.0", "layer.1", "layer.2", "layer.3"),
    )

    assert groups.num_engine_kv_cache_groups == 2
    assert groups.per_layer_engine_group_indices(4) == [0, 1, 0, 1]


def test_vllm_kv_cache_groups_inflates_by_lmcache_layer_identity():
    groups = inflated_lmcache_kv_cache_groups_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0", "layer.2", "layer.4"]),
                MockKVCacheGroup(["layer.1", "layer.3"]),
            ]
        ),
        {
            "layer.0": torch.randn(2, 32, 16, 8, 64, dtype=torch.float16),
            "layer.1": torch.randn(2, 32, 16, 8, 64, dtype=torch.float16),
            "layer.2": torch.randn(2, 32, 16, 8, 64, dtype=torch.float16),
            "layer.3": torch.randn(2, 32, 16, 8, 64, dtype=torch.float16),
            "layer.4": torch.randn(2, 32, 16, 16, 64, dtype=torch.float16),
        },
    )

    assert [group.engine_kv_cache_group_id for group in groups.groups] == [0, 1, 0]
    assert [group.layer_indices for group in groups.groups] == [(0, 2), (1, 3), (4,)]
    assert groups.expand_engine_block_ids_to_lmc_groups([[10], [20]]) == [
        [10],
        [20],
        [10],
    ]
