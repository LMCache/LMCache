# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
)
from lmcache.v1.multiprocess.group_view import (
    expand_engine_block_ids,
    get_engine_group_indices,
    num_engine_groups,
)


@dataclass
class MockKVCacheSpec:
    block_size: int


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]
    kv_cache_spec: MockKVCacheSpec


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


def _same_shape_caches(names: list[str]) -> dict[str, torch.Tensor]:
    return {n: torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for n in names}


def test_conversion_defaults_to_single_group_without_config():
    """No vLLM KV cache groups -> all layers fall into a single engine group."""
    spec = create_engine_group_infos_from_vllm(
        None, _same_shape_caches(["layer.0", "layer.1"])
    )

    assert num_engine_groups(spec) == 1
    assert [group.engine_group_id for group in spec] == [0]
    assert spec[0].layer_indices == (0, 1)


def test_conversion_preserves_engine_group_layers():
    """Two engine groups with identical tensor shape stay separate by group."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.2"], MockKVCacheSpec(block_size=16)
                ),
                MockKVCacheGroup(
                    ["layer.1", "layer.3"], MockKVCacheSpec(block_size=16)
                ),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"]),
    )

    assert num_engine_groups(spec) == 2
    assert get_engine_group_indices(spec, 4) == [0, 1, 0, 1]
    assert [group.tokens_per_block for group in spec] == [16, 16]


def test_conversion_splits_by_lmcache_layer_identity():
    """Layers split by both engine group and physical transfer identity."""
    caches = _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"])
    # layer.4 has a different head count -> distinct transfer identity.
    caches["layer.4"] = torch.randn(2, 32, 16, 16, 64, dtype=torch.float16)
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.2", "layer.4"], MockKVCacheSpec(block_size=16)
                ),
                MockKVCacheGroup(
                    ["layer.1", "layer.3"], MockKVCacheSpec(block_size=16)
                ),
            ]
        ),
        caches,
    )

    assert [group.engine_group_id for group in spec] == [0, 1, 0]
    assert [group.layer_indices for group in spec] == [(0, 2), (1, 3), (4,)]
    assert expand_engine_block_ids(spec, [[10], [20]]) == [
        [10],
        [20],
        [10],
    ]
