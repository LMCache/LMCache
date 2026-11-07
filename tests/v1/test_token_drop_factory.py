# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental import token_drop
from lmcache.integration.vllm.experimental.token_drop import (
    TokenDropSpec,
    build_token_drop_algorithm,
)


def test_factory_instantiates_selected_plugin_with_request_config(monkeypatch):
    configs = []

    def factory(config):
        configs.append(config)
        return object()

    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [
            SimpleNamespace(name="other", load=lambda: lambda _config: None),
            SimpleNamespace(name="rkv", load=lambda: factory),
        ],
    )
    config = {"budget": 128}
    first = build_token_drop_algorithm(TokenDropSpec("rkv", config))
    second = build_token_drop_algorithm(TokenDropSpec("rkv", config))

    assert first is not second
    assert configs == [{"budget": 128}, {"budget": 128}]
    assert all(item is not config for item in configs)


def test_factory_rejects_missing_plugin(monkeypatch):
    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: [])
    with pytest.raises(ValueError, match="No token-drop algorithm plugin"):
        build_token_drop_algorithm(TokenDropSpec("rkv", {}))


def test_factory_rejects_duplicate_plugin(monkeypatch):
    plugin = SimpleNamespace(name="rkv", load=lambda: lambda config: object())
    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: [plugin, plugin])
    with pytest.raises(ValueError, match="Multiple token-drop algorithm plugins"):
        build_token_drop_algorithm(TokenDropSpec("rkv", {}))
