# SPDX-License-Identifier: Apache-2.0
"""Tests for validation of vLLM KV cache group compatibility."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="KV cache group validation imports vLLM")

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    validate_kv_cache_groups,
)


@dataclass
class FullAttentionSpec:
    block_size: int = 16


@dataclass
class CircularBufferSpec:
    block_size: int = 8


@dataclass
class DerivedCircularBufferSpec(CircularBufferSpec):
    pass


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int = 16
    kv_cache_specs: dict[str, object] = field(default_factory=dict)


def _config(*specs: object) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    )


def test_validate_kv_cache_groups_allows_regular_attention() -> None:
    validate_kv_cache_groups(_config(FullAttentionSpec()))


def test_validate_kv_cache_groups_rejects_circular_buffer() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"group 1: CircularBufferSpec .*"
            r"request-lifetime ring block cannot provide per-chunk snapshots"
        ),
    ):
        validate_kv_cache_groups(_config(FullAttentionSpec(), CircularBufferSpec()))


def test_validate_kv_cache_groups_rejects_wrapped_circular_buffer() -> None:
    wrapped = UniformTypeKVCacheSpecs(
        kv_cache_specs={
            "attention": FullAttentionSpec(),
            "compressor": DerivedCircularBufferSpec(),
        }
    )

    with pytest.raises(ValueError, match=r"group 0: CircularBufferSpec"):
        validate_kv_cache_groups(_config(wrapped))
