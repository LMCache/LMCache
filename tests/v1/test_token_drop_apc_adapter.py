# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.token_drop_apc_adapter import _cache_private_or_normal


def _request(spec=None):
    return SimpleNamespace(kv_transfer_params=spec)


def test_normal_request_calls_original_without_changes():
    seen = []

    def original(*args):
        seen.append(args)

    pool = object()
    request = _request({"lmcache.max_offload_tokens": 32})
    blocks = [object()]
    _cache_private_or_normal(original, pool, request, blocks, 0, 1, 16, 0)
    assert seen == [(pool, request, blocks, 0, 1, 16, 0, None)]


def test_td_request_never_publishes_blocks():
    def original(*args):
        pytest.fail("TD block must not be published")

    _cache_private_or_normal(
        original,
        object(),
        _request({"lmcache.token_drop": {"algorithm": "rkv", "config": {}}}),
        [object()],
        0,
        1,
        16,
        0,
    )


def test_invalid_td_config_fails_closed():
    with pytest.raises(ValueError):
        _cache_private_or_normal(
            lambda *args: None,
            object(),
            _request({"lmcache.token_drop": {}}),
            [],
            0,
            1,
            16,
            0,
        )
