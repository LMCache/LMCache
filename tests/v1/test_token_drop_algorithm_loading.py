# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import token_drop
from lmcache.integration.vllm.token_drop import TokenDropSpec


def test_factory_uses_matching_plugin_and_copies_config(monkeypatch):
    configs = []
    plugin = SimpleNamespace(
        load=lambda: lambda config: configs.append(config) or object()
    )

    class _Entries:
        def select(self, name):
            return [plugin] if name == "rkv" else []

    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: _Entries())
    config = {"budget": 8}
    first = token_drop.build_token_drop_algorithm(TokenDropSpec("rkv", config))
    second = token_drop.build_token_drop_algorithm(TokenDropSpec("rkv", config))
    assert first is not second
    assert configs == [config, config]
    assert all(c is not config for c in configs)


def test_unregistered_algorithm_fails_closed(monkeypatch):
    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: SimpleNamespace(select=lambda *, name: []),
    )
    with pytest.raises(ValueError, match="Expected one"):
        token_drop.build_token_drop_algorithm(TokenDropSpec("unknown", {}))
