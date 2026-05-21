# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Buildkite CacheBlend disagg proxy telemetry gate."""

# Standard
import importlib.util
from pathlib import Path


_PROXY_PATH = (
    Path(__file__).parents[2]
    / ".buildkite"
    / "k3_tests"
    / "blend"
    / "scripts"
    / "proxy.py"
)


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("blend_proxy_under_test", _PROXY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pending_requests.clear()
    module.pending_request_aliases.clear()
    module.unmatched_tp_arrivals.clear()
    module.pending_tp_state.clear()
    module.main_event_loop = None
    return module


def test_response_id_alias_unblocks_pending_request():
    proxy = _load_proxy_module()
    event = proxy.create_pending_request("proxy-req-1")

    assert proxy.notify_request("cmpl-vllm-response-1") is False
    assert not event.is_set()

    proxy.register_pending_request_alias("proxy-req-1", "cmpl-vllm-response-1")

    assert event.is_set()
    assert "cmpl-vllm-response-1" not in proxy.unmatched_tp_arrivals


def test_response_id_alias_replays_all_tp_worker_arrivals():
    proxy = _load_proxy_module()
    event = proxy.create_pending_request("proxy-req-2")

    assert (
        proxy.notify_request("chatcmpl-vllm-response-2", world_size=2, kv_rank=0)
        is False
    )
    assert (
        proxy.notify_request("chatcmpl-vllm-response-2", world_size=2, kv_rank=1)
        is False
    )
    assert not event.is_set()

    proxy.register_pending_request_alias("proxy-req-2", "chatcmpl-vllm-response-2")

    assert event.is_set()
    assert proxy.pending_tp_state == {}


def test_remove_pending_request_removes_response_id_alias():
    proxy = _load_proxy_module()
    proxy.create_pending_request("proxy-req-3")
    proxy.register_pending_request_alias("proxy-req-3", "cmpl-vllm-response-3")

    proxy.remove_pending_request("proxy-req-3")

    assert proxy.pending_request_aliases == {}
    assert proxy._resolve_pending_request_id("cmpl-vllm-response-3") is None
