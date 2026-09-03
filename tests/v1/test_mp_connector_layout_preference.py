# SPDX-License-Identifier: Apache-2.0
"""Unit tests for connector_cls.get_required_kvcache_layout."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest


@pytest.fixture
def connector_cls():
    """Import lazily: the connector module initializes process-wide MP
    transfer state at import time, which contaminates later cache-server
    tests when imported at collection. Skips where vLLM is not installed
    (e.g. the LMCache-only unit environment)."""
    module = pytest.importorskip(
        "lmcache.integration.vllm.lmcache_mp_connector",
        reason="requires vLLM",
    )
    return module.LMCacheMPConnector


class LegacyCacheConfig:
    """Cache config without the resolved-layout accessor (pre-vllm#51718)."""


def make_config(use_mla: bool = False, legacy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(use_mla=use_mla),
        cache_config=(
            LegacyCacheConfig()
            if legacy
            else SimpleNamespace(get_resolved_kv_cache_layout=lambda: None)
        ),
    )


def test_non_mla_defers_to_vllm(connector_cls):
    assert connector_cls.get_required_kvcache_layout(make_config()) is None


def test_mla_defers_to_vllm(connector_cls):
    assert connector_cls.get_required_kvcache_layout(make_config(use_mla=True)) is None


def test_legacy_vllm_gets_legacy_name(connector_cls):
    assert connector_cls.get_required_kvcache_layout(make_config(legacy=True)) is None


def test_missing_model_config_defers(connector_cls):
    config = SimpleNamespace(model_config=None, cache_config=SimpleNamespace())
    assert connector_cls.get_required_kvcache_layout(config) is None
