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


@pytest.mark.parametrize(
    ("plugin_names", "expected_error"),
    [
        pytest.param([], "No token-drop algorithm plugin", id="missing-plugin"),
        pytest.param(
            ["rkv", "rkv"],
            "Multiple token-drop algorithm plugins",
            id="duplicate-plugin",
        ),
    ],
)
def test_factory_rejects_missing_or_duplicate_plugin(
    monkeypatch, plugin_names, expected_error
):
    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [
            SimpleNamespace(name=name, load=lambda: lambda _config: object())
            for name in plugin_names
        ],
    )
    with pytest.raises(ValueError, match=expected_error):
        build_token_drop_algorithm(TokenDropSpec("rkv", {}))
