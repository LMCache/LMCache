# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm import utils


def test_vllm_layout_hints_includes_layerwise_when_enabled(monkeypatch):
    monkeypatch.setattr(utils, "try_get_vllm_kv_cache_layout", lambda: "HND")
    monkeypatch.setattr(
        utils,
        "lmcache_get_or_create_config",
        lambda: SimpleNamespace(use_layerwise=True),
    )

    assert utils.vllm_layout_hints() == {
        "kv_layout": "HND",
        "use_layerwise": True,
    }


def test_vllm_layout_hints_omits_layerwise_when_disabled(monkeypatch):
    monkeypatch.setattr(utils, "try_get_vllm_kv_cache_layout", lambda: None)
    monkeypatch.setattr(
        utils,
        "lmcache_get_or_create_config",
        lambda: SimpleNamespace(use_layerwise=False),
    )

    assert utils.vllm_layout_hints() == {}


def test_vllm_layout_hints_accepts_explicit_config(monkeypatch):
    monkeypatch.setattr(utils, "try_get_vllm_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(
        utils,
        "lmcache_get_or_create_config",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected config load")),
    )

    hints = utils.vllm_layout_hints(SimpleNamespace(use_layerwise=True))

    assert hints == {
        "kv_layout": "NHD",
        "use_layerwise": True,
    }
