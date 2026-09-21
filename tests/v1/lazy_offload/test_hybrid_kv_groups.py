# SPDX-License-Identifier: Apache-2.0
"""Pure retention-contract tests for hybrid vLLM KV-cache groups."""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    KVGroupRetentionKind,
    create_kv_group_retention_specs,
    get_tokens_per_block,
)


@dataclass
class AttentionSpec:
    block_size: int


@dataclass
class FullAttentionSpec(AttentionSpec):
    pass


@dataclass
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int


@dataclass
class MambaSpec:
    block_size: int
    mamba_cache_mode: str = "align"


@dataclass
class CircularBufferSpec:
    block_size: int
    prefix_cacheable: bool = False


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict[str, object] = field(default_factory=dict)


@dataclass
class MockKVCacheGroup:
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


def test_hybrid_groups_are_bucketed_by_retention_contract() -> None:
    config = MockKVCacheConfig(
        [
            MockKVCacheGroup(FullAttentionSpec(16)),
            MockKVCacheGroup(SlidingWindowSpec(16, 384)),
            MockKVCacheGroup(SlidingWindowSpec(16, 400)),
            MockKVCacheGroup(MambaSpec(16)),
            MockKVCacheGroup(CircularBufferSpec(8)),
        ]
    )

    specs = create_kv_group_retention_specs(
        config,
        [16, 16, 16, 16, 0],
        lmcache_tokens_per_chunk=256,
    )

    assert [spec.kind for spec in specs] == [
        KVGroupRetentionKind.FULL_ATTENTION,
        KVGroupRetentionKind.SLIDING_WINDOW,
        KVGroupRetentionKind.SLIDING_WINDOW,
        KVGroupRetentionKind.RECURRENT,
        KVGroupRetentionKind.SCRATCH,
    ]
    assert [spec.retention_group_id for spec in specs] == [0, 1, 1, 2, None]
    assert [spec.cacheable for spec in specs] == [True, True, True, True, False]


def test_dcp_scales_attention_blocks_but_not_recurrent_or_scratch() -> None:
    assert get_tokens_per_block(FullAttentionSpec(16), dcp_size=4) == 64
    assert get_tokens_per_block(MambaSpec(16), dcp_size=4) == 16
    assert get_tokens_per_block(CircularBufferSpec(8), dcp_size=4) == 0


def test_one_engine_group_cannot_mix_retention_contracts() -> None:
    mixed = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "full": FullAttentionSpec(16),
            "sliding": SlidingWindowSpec(16, 128),
        },
    )

    with pytest.raises(ValueError, match="incompatible retention contracts"):
        create_kv_group_retention_specs(
            MockKVCacheConfig([MockKVCacheGroup(mixed)]),
            [16],
            lmcache_tokens_per_chunk=256,
        )
