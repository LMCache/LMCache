# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vLLM KV-cache layout discovery and normalization.

Covers the post-vllm#51718 world (layout resolved once onto ``CacheConfig``),
the legacy attention-backend query, and the fail-loud contract for
standardized layouts LMCache cannot transfer yet. vLLM is stubbed through
``sys.modules``; no GPU or vLLM install needed.
"""

# Standard
from types import ModuleType, SimpleNamespace
import sys

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import (
    try_get_vllm_kv_cache_layout,
    vllm_layout_hints,
)

UNSUPPORTED_LAYOUTS = ("LHBNC", "BHLNC")


def make_vllm_config(resolved_layout_name: str) -> SimpleNamespace:
    """Fake config whose accessor returns a resolved layout member name."""
    member = SimpleNamespace(name=resolved_layout_name)
    cache_config = SimpleNamespace(get_resolved_kv_cache_layout=lambda: member)
    return SimpleNamespace(cache_config=cache_config)


def make_unresolved_vllm_config() -> SimpleNamespace:
    def _raise() -> SimpleNamespace:
        raise ValueError("KV cache layout has not been resolved yet")

    return SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=_raise)
    )


class LegacyCacheConfig:
    """Cache config without the resolved-layout accessor (pre-vllm#51718)."""


def stub_module(
    monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object
) -> ModuleType:
    """Install a fake module (and empty parent packages) into ``sys.modules``."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
    monkeypatch.setitem(sys.modules, name, module)
    return module


class TestTryGetVllmKVCacheLayout:
    def test_resolved_member_name_is_the_hint(self):
        config = make_vllm_config("LBNHC")
        assert try_get_vllm_kv_cache_layout(config) == "LBNHC"

    def test_unresolved_layout_error_passes_through(self):
        with pytest.raises(ValueError, match="resolved"):
            try_get_vllm_kv_cache_layout(make_unresolved_vllm_config())

    @pytest.mark.parametrize("name", UNSUPPORTED_LAYOUTS)
    def test_unsupported_resolved_layout_raises(self, name):
        config = make_vllm_config(name)
        with pytest.raises(NotImplementedError, match=name):
            try_get_vllm_kv_cache_layout(config)

    def test_ambient_config_is_used_when_not_passed(self, monkeypatch):
        config = make_vllm_config("LBHNC")
        stub_module(monkeypatch, "vllm.config", get_current_vllm_config=lambda: config)
        assert try_get_vllm_kv_cache_layout() == "LBHNC"

    def test_legacy_vllm_falls_back_to_backend_query(self, monkeypatch):
        stub_module(
            monkeypatch,
            "vllm.v1.attention.backends.utils",
            get_kv_cache_layout=lambda: "HND",
        )
        config = SimpleNamespace(cache_config=LegacyCacheConfig())
        assert try_get_vllm_kv_cache_layout(config) == "HND"

    def test_vllm_absent_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vllm", None)
        monkeypatch.setitem(sys.modules, "vllm.config", None)
        assert try_get_vllm_kv_cache_layout() is None


class TestVllmLayoutHints:
    def test_hint_carries_the_member_name(self):
        config = make_vllm_config("LBHNC")
        assert vllm_layout_hints(config) == {"kv_layout": "LBHNC"}

    def test_unresolved_layout_raises_through_hints(self):
        with pytest.raises(ValueError, match="resolved"):
            vllm_layout_hints(make_unresolved_vllm_config())


class TestResolveVllmKVLayout:
    @pytest.fixture
    def resolve(self):
        detectors_vllm = pytest.importorskip(
            "lmcache.v1.gpu_connector.kv_format.detectors.vllm"
        )
        return detectors_vllm.resolve_vllm_kv_layout

    def test_cpu_backend_forces_head_contiguous(self, resolve):
        assert resolve({"kv_layout": "NHD"}, cpu_attention_backend=True) == "HND"

    def test_missing_hint_defaults_to_nhd(self, resolve):
        assert resolve({}, cpu_attention_backend=False) == "NHD"

    def test_hnd_hint_passes_through(self, resolve):
        assert resolve({"kv_layout": "HND"}, cpu_attention_backend=False) == "HND"

    def test_unsupported_hint_rejected(self, resolve):
        with pytest.raises(ValueError, match="LHBNC"):
            resolve({"kv_layout": "LHBNC"}, cpu_attention_backend=False)
