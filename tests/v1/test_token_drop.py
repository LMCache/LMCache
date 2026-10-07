# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from lmcache.integration.vllm import token_drop
from lmcache.integration.vllm.token_drop import (
    TokenDropSpec,
    build_token_drop_algorithm,
    parse_token_drop_spec,
)


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_absent_token_drop_config_means_normal_request() -> None:
    assert parse_token_drop_spec(None) is None
    assert parse_token_drop_spec({}) is None
    assert parse_token_drop_spec({"lmcache.max_offload_tokens": 64}) is None


def test_parse_request_local_algorithm_config_is_opaque_to_lmcache() -> None:
    config = {
        "algorithm_owned_knob": {"nested": True},
        "future_knob": 17,
    }
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "fake",
                "config": config,
            }
        }
    )

    assert spec == TokenDropSpec(
        algorithm="fake",
        config=config,
    )


def test_algorithm_factory_uses_external_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def factory(config):
        seen.update(config)
        return SimpleNamespace(name="fake")

    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [_FakeEntryPoint("fake", factory)],
    )

    algorithm = build_token_drop_algorithm(
        TokenDropSpec(
            algorithm="fake",
            config={"opaque": 1},
        )
    )
    assert algorithm.name == "fake"
    assert seen == {"opaque": 1}


def test_missing_algorithm_plugin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: [])

    with pytest.raises(ValueError, match="No token-drop algorithm plugin"):
        build_token_drop_algorithm(TokenDropSpec("missing", {}))


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("fake", "JSON object"),
        ({"config": {}}, "algorithm"),
        ({"algorithm": "fake"}, "config"),
        ({"algorithm": "fake", "config": []}, "config"),
        (
            {
                "algorithm": "fake",
                "config": {},
                "unexpected": True,
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_token_drop_envelope_fails_closed(raw, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_token_drop_spec({"lmcache.token_drop": raw})
