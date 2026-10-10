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
    get_kv_dtype_decorated_model_name,
)


def name(cache: CacheConfig, model_dtype: torch.dtype = torch.bfloat16) -> str:
    """Compute a cache name using validated vLLM cache configuration."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=model_dtype), cache_config=cache
    )
    return get_kv_dtype_decorated_model_name(cast(VllmConfig, config), "org/model")


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
    assert name(left) != name(right)


def test_same_precision_shares_across_cache_capacities() -> None:
    assert name(CacheConfig(gpu_memory_utilization=0.7)) == name(
        CacheConfig(gpu_memory_utilization=0.9)
    )
    assert name(CacheConfig(), torch.float16) != name(CacheConfig(), torch.bfloat16)
