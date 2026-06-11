# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
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


def _full_attention_spec(sliding_window: "int | None" = None) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )


def _sliding_window_spec(sliding_window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )


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


def test_conversion_resolves_sliding_window_size():
    """A SlidingWindowSpec group carries its window size in tokens."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0"], _full_attention_spec()),
                MockKVCacheGroup(["layer.1"], _sliding_window_spec(64)),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1, 64]


def test_conversion_ignores_full_attention_sliding_window():
    """SWA layers managed as full attention (hybrid allocator disabled) are
    not sliding window: vLLM allocates blocks for all tokens."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.1"], _full_attention_spec(sliding_window=1024)
                ),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1]


def test_conversion_defaults_sliding_window_for_non_sw_spec():
    """Groups whose spec is not a SlidingWindowSpec resolve to
    non-sliding-window."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0"], MockKVCacheSpec(block_size=16))
            ]
        ),
        _same_shape_caches(["layer.0"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1]


def test_conversion_uniform_type_specs_resolve_per_layer():
    """Inside a UniformTypeKVCacheSpecs group, per-layer specs decide the
    window. SW layers with a distinct transfer identity get their own group
    carrying the window size."""
    caches = _same_shape_caches(["layer.0", "layer.1"])
    # layer.1 has a different head count -> distinct transfer identity.
    caches["layer.1"] = torch.randn(2, 32, 16, 16, 64, dtype=torch.float16)
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": _full_attention_spec(),
            "layer.1": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=16,
                head_size=64,
                dtype=torch.float16,
                sliding_window=512,
            ),
        },
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["layer.0", "layer.1"], uniform_spec)]
        ),
        caches,
    )

    assert [group.layer_indices for group in spec] == [(0,), (1,)]
    assert [group.sw_size_tokens for group in spec] == [-1, 512]


def test_conversion_mixed_window_layers_fall_back_to_full_attention():
    """Same-identity layers mixing different windows conservatively resolve
    to non-sliding-window for the merged group."""
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": _full_attention_spec(),
            "layer.1": _sliding_window_spec(64),
        },
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["layer.0", "layer.1"], uniform_spec)]
        ),
        _same_shape_caches(["layer.0", "layer.1"]),
    )

    assert [group.layer_indices for group in spec] == [(0, 1)]
    assert [group.sw_size_tokens for group in spec] == [-1]
