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
    translate_vllm_kv_cache_layout,
    try_get_vllm_kv_cache_layout,
    vllm_layout_hints,
)

UNSUPPORTED_LAYOUTS = ("LHBNC", "BLHNC", "BLNHC", "BHLNC")


def make_vllm_config(**cache_attrs: object) -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(**cache_attrs))


class LegacyCacheConfig:
    """Cache config without ``kv_cache_layout`` (pre-vllm#51718)."""


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


class TestTranslateVllmKVCacheLayout:
    def test_legacy_names_pass_through(self):
        assert translate_vllm_kv_cache_layout("NHD") == "NHD"
        assert translate_vllm_kv_cache_layout("HND") == "HND"

    def test_layer_compact_names_map_to_hint_vocabulary(self):
        assert translate_vllm_kv_cache_layout("LBNHC") == "NHD"
        assert translate_vllm_kv_cache_layout("LBHNC") == "HND"

    @pytest.mark.parametrize("name", UNSUPPORTED_LAYOUTS)
    def test_unsupported_layouts_fail_loudly(self, name):
        with pytest.raises(NotImplementedError, match=name):
            translate_vllm_kv_cache_layout(name)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="XYZ"):
            translate_vllm_kv_cache_layout("XYZ")


class TestTryGetVllmKVCacheLayout:
    def test_resolved_layout_from_explicit_config(self):
        config = make_vllm_config(kv_cache_layout="LBNHC")
        assert try_get_vllm_kv_cache_layout(config) == "NHD"

    def test_stored_alias_from_explicit_config(self):
        config = make_vllm_config(kv_cache_layout="HND")
        assert try_get_vllm_kv_cache_layout(config) == "HND"

    def test_unresolved_layout_fails_loudly(self):
        config = make_vllm_config(kv_cache_layout=None)
        with pytest.raises(ValueError, match="not resolved"):
            try_get_vllm_kv_cache_layout(config)

    @pytest.mark.parametrize("name", UNSUPPORTED_LAYOUTS)
    def test_unsupported_resolved_layout_raises(self, name):
        config = make_vllm_config(kv_cache_layout=name)
        with pytest.raises(NotImplementedError, match=name):
            try_get_vllm_kv_cache_layout(config)

    def test_ambient_config_is_used_when_not_passed(self, monkeypatch):
        config = make_vllm_config(kv_cache_layout="LBHNC")
        stub_module(monkeypatch, "vllm.config", get_current_vllm_config=lambda: config)
        assert try_get_vllm_kv_cache_layout() == "HND"

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
    def test_hint_present_and_normalized(self):
        config = make_vllm_config(kv_cache_layout="LBHNC")
        assert vllm_layout_hints(config) == {"kv_layout": "HND"}

    def test_unresolved_layout_raises_through_hints(self):
        config = make_vllm_config(kv_cache_layout=None)
        with pytest.raises(ValueError, match="not resolved"):
            vllm_layout_hints(config)


class TestResolveVllmKVLayout:
    @pytest.fixture
    def resolve(self):
        detectors_vllm = pytest.importorskip(
            "lmcache.v1.gpu_connector.kv_format.detectors.vllm"
        )
        return detectors_vllm.resolve_vllm_kv_layout

    def test_cpu_backend_forces_hnd(self, resolve):
        assert resolve({"kv_layout": "NHD"}, cpu_attention_backend=True) == "HND"

    def test_missing_hint_defaults_to_nhd(self, resolve):
        assert resolve({}, cpu_attention_backend=False) == "NHD"

    def test_hint_passes_through(self, resolve):
        assert resolve({"kv_layout": "HND"}, cpu_attention_backend=False) == "HND"

    def test_untranslated_hint_rejected(self, resolve):
        with pytest.raises(ValueError, match="LBHNC"):
            resolve({"kv_layout": "LBHNC"}, cpu_attention_backend=False)
