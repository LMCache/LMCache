# SPDX-License-Identifier: Apache-2.0
"""Encoding namespaces separate incompatible opaque KV bytes."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.config import CacheConfig, VllmConfig  # noqa: E402

# First Party
from lmcache.integration.vllm.kv_cache_identity import (  # noqa: E402
    get_kv_cache_format,
)
from lmcache.v1.cache_format import KVCacheFormat  # noqa: E402


def cache_format(cache: CacheConfig, model_dtype: torch.dtype = torch.bfloat16) -> str:
    """Compute a cache format using validated vLLM cache configuration."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=model_dtype), cache_config=cache
    )
    return get_kv_cache_format(cast(VllmConfig, config))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (CacheConfig(), CacheConfig(cache_dtype="fp8")),
        (CacheConfig(cache_dtype="fp8_e4m3"), CacheConfig(cache_dtype="fp8_e5m2")),
        (CacheConfig(), CacheConfig(mamba_cache_dtype="float32")),
        (CacheConfig(), CacheConfig(mamba_ssm_cache_dtype="float32")),
        (CacheConfig(), CacheConfig(kv_cache_dtype_skip_layers=["0"])),
    ],
)
def test_different_precision_settings_are_isolated(
    left: CacheConfig, right: CacheConfig
) -> None:
    assert cache_format(left) != cache_format(right)


def test_same_precision_shares_across_cache_capacities() -> None:
    assert cache_format(CacheConfig(gpu_memory_utilization=0.7)) == cache_format(
        CacheConfig(gpu_memory_utilization=0.9)
    )
    assert cache_format(CacheConfig(), torch.float16) != cache_format(
        CacheConfig(), torch.bfloat16
    )


def test_vllm_uses_engine_neutral_format_identity() -> None:
    assert (
        cache_format(CacheConfig())
        == KVCacheFormat(
            model_dtype="torch.bfloat16",
            cache_dtype="auto",
            skip_layers=(),
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ).to_cache_format()
    )


def test_skip_layer_order_and_duplicates_do_not_change_identity() -> None:
    assert cache_format(CacheConfig(kv_cache_dtype_skip_layers=["2", "0", "2"])) == (
        cache_format(CacheConfig(kv_cache_dtype_skip_layers=["0", "2"]))
    )
