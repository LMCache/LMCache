# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    create_group_views_from_vllm,
    per_layer_storage_blocks_per_chunk_from_vllm,
    storage_blocks_per_chunk,
)
from lmcache.v1.multiprocess.group_view import (
    expand_block_ids_to_views,
    get_engine_group_indices,
    num_engine_groups,
)


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]

@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


def _same_shape_caches(names: list[str]) -> dict[str, torch.Tensor]:
    return {n: torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for n in names}


def test_conversion_defaults_to_single_group_without_config():
    """No vLLM KV cache groups -> all layers fall into a single engine group."""
    spec = create_group_views_from_vllm(
        None, _same_shape_caches(["layer.0", "layer.1"])
    )

    assert num_engine_groups(spec) == 1
    assert [group.engine_group_id for group in spec] == [0]
    assert spec[0].layer_indices == (0, 1)


def test_conversion_preserves_engine_group_layers():
    """Two engine groups with identical tensor shape stay separate by group."""
    spec = create_group_views_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0", "layer.2"]),
                MockKVCacheGroup(["layer.1", "layer.3"]),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"]),
    )

    assert num_engine_groups(spec) == 2
    assert get_engine_group_indices(spec, 4) == [0, 1, 0, 1]


def test_conversion_splits_by_lmcache_layer_identity():
    """Layers split by both engine group and physical transfer identity."""
    caches = _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"])
    # layer.4 has a different head count -> distinct transfer identity.
    caches["layer.4"] = torch.randn(2, 32, 16, 16, 64, dtype=torch.float16)
    spec = create_group_views_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0", "layer.2", "layer.4"]),
                MockKVCacheGroup(["layer.1", "layer.3"]),
            ]
        ),
        caches,
    )

    assert [group.engine_group_id for group in spec] == [0, 1, 0]
    assert [group.layer_indices for group in spec] == [(0, 2), (1, 3), (4,)]
    assert expand_block_ids_to_views(spec, [[10], [20]]) == [
        [10],
        [20],
        [10],
    ]


# ----------------------------------------------------------------------------
# per_layer_* hint derivation (logical block size + sliding window)
# ----------------------------------------------------------------------------


@dataclass
class MockSpec:
    """A vLLM ``KVCacheGroupSpec.kv_cache_spec`` stand-in."""

    block_size: int = 0
    sliding_window: int | None = None


@dataclass
class MockUniformSpec:
    """A ``UniformTypeKVCacheSpecs`` stand-in: block size/window on inner specs."""

    kv_cache_specs: dict


@dataclass
class MockGroupWithSpec:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockConfigWithSpecs:
    kv_cache_groups: list


def test_storage_blocks_per_chunk_helper():
    """Shared SWA-suffix formula: full chunk vs trailing window, capped."""
    # full attention -> full chunk (1024 / 64 = 16)
    assert storage_blocks_per_chunk(0, 64, 1024) == 16
    # window 128, logical 64 -> ceil(128/64) = 2
    assert storage_blocks_per_chunk(128, 64, 1024) == 2
    # window 8, logical 4 -> ceil(8/4) = 2
    assert storage_blocks_per_chunk(8, 4, 1024) == 2
    # window larger than a chunk -> capped at full chunk
    assert storage_blocks_per_chunk(4096, 64, 1024) == 16
    # compressed MLA-like: logical 256 -> 1024/256 = 4
    assert storage_blocks_per_chunk(0, 256, 1024) == 4


def test_per_layer_storage_blocks_maps_per_group():
    """Each layer gets its group's stored block count (full or SWA suffix)."""
    config = MockConfigWithSpecs(
        kv_cache_groups=[
            # full attention, logical 256 -> full chunk 1024/256 = 4
            MockGroupWithSpec(["layer.0"], MockSpec(block_size=256, sliding_window=0)),
            # SWA, logical 64, window 128 -> ceil(128/64) = 2
            MockGroupWithSpec(["layer.1"], MockSpec(block_size=64, sliding_window=128)),
        ]
    )
    caches = {"layer.0": None, "layer.1": None}
    result = per_layer_storage_blocks_per_chunk_from_vllm(config, caches, 1024)
    assert result == [4, 2]


def test_per_layer_storage_blocks_uniform_inner_spec_fallback():
    """Block size / window read from inner specs for a UniformTypeKVCacheSpecs."""
    uniform = MockUniformSpec(
        kv_cache_specs={"layer.0": MockSpec(block_size=4, sliding_window=8)}
    )
    config = MockConfigWithSpecs(
        kv_cache_groups=[MockGroupWithSpec(["layer.0", "layer.1"], uniform)]
    )
    caches = {"layer.0": None, "layer.1": None}
    result = per_layer_storage_blocks_per_chunk_from_vllm(config, caches, 1024)
    # logical 4, window 8 -> ceil(8/4) = 2 for both layers
    assert result == [2, 2]


def test_per_layer_storage_blocks_none_without_groups():
    """No engine groups -> None (server treats every group as full chunk)."""
    assert per_layer_storage_blocks_per_chunk_from_vllm(None, {}, 1024) is None

