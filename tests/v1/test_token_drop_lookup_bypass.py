# SPDX-License-Identifier: Apache-2.0
"""Per-request external KV lookup bypass for token dropping."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.experimental.token_drop import TokenDropSpec  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)


def make_connector(spec):
    tracker = SimpleNamespace(
        token_drop_spec=spec,
        lookup_started_at=42.0,
        cache_salt="",
        request_configs={},
        get_token_ids=lambda: [1, 2],
    )
    adapter = MagicMock()
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _eager_prefetch=True,
        _reserve_last_token_for_lookup=False,
        _get_or_create_request_tracker=lambda request: tracker,
        scheduler_adapter=adapter,
    )
    request = SimpleNamespace(request_id="req", resumable=False)
    return connector, request, adapter


def test_token_drop_skips_eager_and_scheduler_lookup() -> None:
    connector, request, adapter = make_connector(TokenDropSpec("rkv", {}))
    LMCacheMPConnector.on_new_request(connector, request)
    assert LMCacheMPConnector.get_num_new_matched_tokens(connector, request, 0) == (
        0,
        False,
    )
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.check_lookup_result.assert_not_called()


def test_normal_request_still_submits_eager_lookup() -> None:
    connector, request, adapter = make_connector(None)
    LMCacheMPConnector.on_new_request(connector, request)
    adapter.maybe_submit_lookup_request.assert_called_once_with(
        "req",
        token_ids=[1, 2],
        cache_salt="",
        request_configs={},
        reserve_last_token=False,
    )


def test_mixed_requests_bypass_only_token_drop_lookup() -> None:
    connector, normal_request, adapter = make_connector(None)
    normal_tracker = connector._get_or_create_request_tracker
    connector._get_or_create_request_tracker = lambda request: (
        SimpleNamespace(token_drop_spec=TokenDropSpec("rkv", {}))
        if request.request_id == "td"
        else normal_tracker(request)
    )
    td_request = SimpleNamespace(request_id="td", resumable=False)

    LMCacheMPConnector.on_new_request(connector, td_request)
    assert LMCacheMPConnector.get_num_new_matched_tokens(connector, td_request, 0) == (
        0,
        False,
    )
    adapter.maybe_submit_lookup_request.assert_not_called()

    LMCacheMPConnector.on_new_request(connector, normal_request)
    adapter.maybe_submit_lookup_request.assert_called_once_with(
        "req",
        token_ids=[1, 2],
        cache_salt="",
        request_configs={},
        reserve_last_token=False,
    )
