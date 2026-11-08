# SPDX-License-Identifier: Apache-2.0
"""Request-local token-drop selection at the MP scheduler boundary."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop import TokenDropSpec  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestTracker,
)


def make_request(config=None):
    return SimpleNamespace(
        request_id="request-1",
        cache_salt=None,
        lora_request=None,
        all_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        mm_features=[],
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": config} if config is not None else None
        ),
    )


def test_normal_request_has_no_algorithm() -> None:
    assert LMCacheMPRequestTracker(make_request()).token_drop_spec is None
    assert (
        LMCacheMPRequestTracker(
            make_request({"lmcache.max_offload_tokens": 8})
        ).token_drop_spec
        is None
    )


def test_token_drop_spec_captured_per_request() -> None:
    settings = {"budget": 32}
    request = make_request(
        {"lmcache.token_drop": {"algorithm": "rkv", "config": settings}}
    )
    tracker = LMCacheMPRequestTracker(request, enable_token_drop=True)
    assert tracker.token_drop_spec == TokenDropSpec("rkv", {"budget": 32})
    assert tracker.token_drop_spec.config is not settings


def test_invalid_spec_rejected_during_request_tracking() -> None:
    request = make_request({"lmcache.token_drop": {"algorithm": "rkv"}})
    with pytest.raises(ValueError, match="lmcache.token_drop.config"):
        LMCacheMPRequestTracker(request, enable_token_drop=True)


def test_connector_assigns_generation_once_per_new_td_request() -> None:
    connector = SimpleNamespace(
        request_trackers={},
        lazy_offload=False,
        _token_drop_enabled=True,
        _token_drop_next_generation=0,
        _apply_revision_tag=lambda tracker: None,
    )
    request = make_request(
        {"lmcache.token_drop": {"algorithm": "rkv", "config": {"budget": 32}}}
    )
    request.status = None
    first = LMCacheMPConnector._get_or_create_request_tracker(connector, request)
    assert first.token_drop_generation == 1
    assert connector._token_drop_next_generation == 1
    assert (
        LMCacheMPConnector._get_or_create_request_tracker(connector, request) is first
    )
    assert connector._token_drop_next_generation == 1
    normal = make_request()
    normal.request_id = "normal"
    normal.status = None
    other = LMCacheMPConnector._get_or_create_request_tracker(connector, normal)
    assert other.token_drop_spec is None
    assert connector._token_drop_next_generation == 1
