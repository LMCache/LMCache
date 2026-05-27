# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

import pytest

# First Party
from lmcache.integration.vllm.hma_utils import (
    build_engine_group_layout_hints,
    get_num_engine_groups,
)


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


def test_get_num_engine_groups_defaults_to_one():
    assert get_num_engine_groups(None) == 1
    assert get_num_engine_groups(MockKVCacheConfig(kv_cache_groups=[])) == 1


def test_build_engine_group_layout_hints():
    config = MockKVCacheConfig(
        kv_cache_groups=[
            MockKVCacheGroup(["layer.0", "layer.2"]),
            MockKVCacheGroup(["layer.1", "layer.3"]),
        ]
    )
    kv_caches = {
        "layer.0": object(),
        "layer.1": object(),
        "layer.2": object(),
        "layer.3": object(),
    }

    assert build_engine_group_layout_hints(config, kv_caches) == {
        "per_layer_engine_group_idx": [0, 1, 0, 1]
    }


def test_build_engine_group_layout_hints_requires_all_layers():
    config = MockKVCacheConfig(
        kv_cache_groups=[MockKVCacheGroup(["layer.0"]), MockKVCacheGroup(["layer.1"])]
    )
    kv_caches = {"layer.0": object(), "layer.2": object()}

    with pytest.raises(ValueError, match="did not cover"):
        build_engine_group_layout_hints(config, kv_caches)
