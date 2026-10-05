# SPDX-License-Identifier: Apache-2.0

import pytest

from lmcache.integration.vllm.token_drop import (
    TokenDropSpec,
    parse_token_drop_spec,
)


def test_absent_token_drop_config_means_normal_request() -> None:
    assert parse_token_drop_spec(None) is None
    assert parse_token_drop_spec({}) is None
    assert parse_token_drop_spec({"lmcache.max_offload_tokens": 64}) is None


def test_parse_request_local_rkv_config_is_opaque_to_lmcache() -> None:
    config = {
        "budget": 1024,
        "buffer": 64,
        "window_size": 8,
        "kernel_size": 7,
    }
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "rkv",
                "config": config,
            }
        }
    )

    assert spec == TokenDropSpec(
        algorithm="rkv",
        config=config,
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("rkv", "JSON object"),
        ({"config": {"budget": 32, "buffer": 16}}, "algorithm"),
        (
            {"algorithm": "other", "config": {"budget": 32, "buffer": 16}},
            "Unsupported",
        ),
        ({"algorithm": "rkv"}, "config"),
        ({"algorithm": "rkv", "config": []}, "config"),
        (
            {
                "algorithm": "rkv",
                "config": {"budget": 32, "buffer": 16},
                "unexpected": True,
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_token_drop_envelope_fails_closed(raw, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_token_drop_spec({"lmcache.token_drop": raw})


def test_lmcache_does_not_validate_rkv_specific_config() -> None:
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "rkv",
                "config": {
                    "budget": "validated-by-rkv",
                    "some_future_rkv_knob": object(),
                },
            }
        }
    )
    assert spec is not None
    assert spec.config["budget"] == "validated-by-rkv"
    assert "some_future_rkv_knob" in spec.config
