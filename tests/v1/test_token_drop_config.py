# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.token_drop import TokenDropSpec, parse_token_drop_spec
from lmcache.integration.vllm.utils import extract_request_configs_from_request


def test_no_token_drop_keeps_normal_request() -> None:
    assert parse_token_drop_spec(None) is None
    assert parse_token_drop_spec({"lmcache.max_offload_tokens": 64}) is None


def test_parse_per_request_algorithm_and_opaque_config() -> None:
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
    assert spec == TokenDropSpec("rkv", config)
    assert spec.config is not config


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"algorithm": "rkv"},
        {"config": {}},
        {"algorithm": "", "config": {}},
        {"algorithm": 1, "config": {}},
        {"algorithm": "rkv", "config": None},
        {"algorithm": "rkv", "config": {}, "extra": True},
    ],
)
def test_invalid_token_drop_config_is_rejected(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_token_drop_spec({"lmcache.token_drop": raw})
