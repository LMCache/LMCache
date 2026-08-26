# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheMPConnector.get_required_kvcache_layout."""

# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector


class LegacyCacheConfig:
    """Cache config without ``kv_cache_layout`` (pre-vllm#51718)."""


def make_config(use_mla: bool = False, legacy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(use_mla=use_mla),
        cache_config=(
            LegacyCacheConfig() if legacy else SimpleNamespace(kv_cache_layout=None)
        ),
    )


def test_non_mla_prefers_standardized_head_contiguous_layout():
    assert LMCacheMPConnector.get_required_kvcache_layout(make_config()) == "LBHNC"


def test_mla_defers_to_vllm():
    assert (
        LMCacheMPConnector.get_required_kvcache_layout(make_config(use_mla=True))
        is None
    )


def test_legacy_vllm_gets_legacy_name():
    assert (
        LMCacheMPConnector.get_required_kvcache_layout(make_config(legacy=True))
        == "HND"
    )


def test_missing_model_config_defers():
    config = SimpleNamespace(model_config=None, cache_config=SimpleNamespace())
    assert LMCacheMPConnector.get_required_kvcache_layout(config) is None
