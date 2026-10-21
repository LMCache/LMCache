# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Any, NamedTuple

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    TokenDropSpec,
    parse_token_drop_spec,
)
from lmcache.integration.vllm.utils import extract_request_configs_from_request


class ParseCase(NamedTuple):
    name: str
    input: dict[str, Any] | None
    expected: TokenDropSpec | None | type[ValueError]


@pytest.mark.parametrize(
    "case",
    [
        ParseCase("no-config", None, None),
        ParseCase("empty-request-config", {}, None),
        ParseCase("normal-request", {"lmcache.max_offload_tokens": 64}, None),
        ParseCase(
            "empty-algorithm-config",
            {"lmcache.token_drop": {"algorithm": "rkv", "config": {}}},
            TokenDropSpec("rkv", {}),
        ),
        ParseCase(
            "opaque-algorithm-config",
            {
                "lmcache.token_drop": {
                    "algorithm": "rkv",
                    "config": {"future_knob": {"nested": True}},
                }
            },
            TokenDropSpec("rkv", {"future_knob": {"nested": True}}),
        ),
        ParseCase("null-envelope", {"lmcache.token_drop": None}, ValueError),
        ParseCase("list-envelope", {"lmcache.token_drop": []}, ValueError),
        ParseCase("empty-envelope", {"lmcache.token_drop": {}}, ValueError),
        ParseCase(
            "missing-config", {"lmcache.token_drop": {"algorithm": "rkv"}}, ValueError
        ),
        ParseCase(
            "missing-algorithm", {"lmcache.token_drop": {"config": {}}}, ValueError
        ),
        ParseCase(
            "empty-algorithm",
            {"lmcache.token_drop": {"algorithm": "", "config": {}}},
            ValueError,
        ),
        ParseCase(
            "non-string-algorithm",
            {"lmcache.token_drop": {"algorithm": 1, "config": {}}},
            ValueError,
        ),
        ParseCase(
            "null-algorithm-config",
            {"lmcache.token_drop": {"algorithm": "rkv", "config": None}},
            ValueError,
        ),
        ParseCase(
            "list-algorithm-config",
            {"lmcache.token_drop": {"algorithm": "rkv", "config": []}},
            ValueError,
        ),
        ParseCase(
            "mixed-type-unknown-keys",
            {
                "lmcache.token_drop": {
                    "algorithm": "rkv",
                    "config": {},
                    7: True,
                    "extra": True,
                }
            },
            ValueError,
        ),
        ParseCase(
            "unexpected-envelope-key",
            {"lmcache.token_drop": {"algorithm": "rkv", "config": {}, "extra": True}},
            ValueError,
        ),
    ],
    ids=lambda case: case.name,
)
def test_parse_token_drop_spec(case: ParseCase) -> None:
    if case.expected is ValueError:
        with pytest.raises(ValueError):
            parse_token_drop_spec(case.input)
    else:
        assert parse_token_drop_spec(case.input) == case.expected


def test_request_extraction_and_config_copy() -> None:
    config = {"future_knob": {"nested": True}}
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": config},
                    "transport_only": "ignored",
                }
            }
        )
    )

    spec = parse_token_drop_spec(extract_request_configs_from_request(request))
    assert spec is not None
    assert spec == TokenDropSpec("rkv", config)
    assert spec.config is not config
