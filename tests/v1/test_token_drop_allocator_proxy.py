# SPDX-License-Identifier: Apache-2.0
"""Engine-local proxy contract. Full vLLM scheduler integration is a separate gate."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


@pytest.fixture
def proxy_factory():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_scheduler_proxy import (
        _EngineLocalKVCacheManagerProxy,
    )

    return _EngineLocalKVCacheManagerProxy


class _FakeManager:
    max_model_len = 2048

    def __init__(self):
        self.calls = []
        self.result = object()
        self.error = False

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        self.calls.append((request, request.num_computed_tokens, num_new_tokens))
        if self.error:
            raise RuntimeError("injected failure")
        return self.result


def _request(request_id: str, logical: int, *, token_drop: bool):
    config = (
        {"lmcache.token_drop": {"algorithm": "rkv", "config": {}}} if token_drop else {}
    )
    return SimpleNamespace(
        request_id=request_id,
        num_computed_tokens=logical,
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": config}),
    )


@pytest.fixture(autouse=True)
def _clear_state():
    for rid in ("td", "normal", "td2"):
        clear_resident_kv_tokens(rid)
    yield
    for rid in ("td", "normal", "td2"):
        clear_resident_kv_tokens(rid)


def test_mixed_requests_only_td_uses_physical_length(proxy_factory):
    native = _FakeManager()
    proxy = proxy_factory(native)
    td = _request("td", 104, token_drop=True)
    normal = _request("normal", 104, token_drop=False)
    set_resident_kv_tokens("td", 32)

    assert proxy.allocate_slots(td, 1) is native.result
    assert native.calls[-1][0] is not td
    assert native.calls[-1][1] == 32
    assert td.num_computed_tokens == 104
    assert get_resident_kv_tokens("td") == 33

    assert proxy.allocate_slots(normal, 1) is native.result
    assert native.calls[-1][0] is normal
    assert native.calls[-1][1] == 104
    assert get_resident_kv_tokens("normal") is None


def test_preemption_and_stale_state_reuse_s2_guards(proxy_factory):
    native = _FakeManager()
    proxy = proxy_factory(native)
    td = _request("td", 0, token_drop=True)
    set_resident_kv_tokens("td", 32)

    proxy.allocate_slots(td, 1)
    assert native.calls[-1][0] is td
    assert native.calls[-1][1] == 0
    assert get_resident_kv_tokens("td") is None

    normal = _request("td", 104, token_drop=False)
    set_resident_kv_tokens("td", 32)
    proxy.allocate_slots(normal, 1)
    assert native.calls[-1][0] is normal
    assert native.calls[-1][1] == 104
    assert get_resident_kv_tokens("td") is None


def test_failure_and_unsupported_mode_leave_frontier_unchanged(proxy_factory):
    native = _FakeManager()
    proxy = proxy_factory(native)
    td = _request("td", 104, token_drop=True)
    set_resident_kv_tokens("td", 32)
    native.result = None
    assert proxy.allocate_slots(td, 1) is None
    assert get_resident_kv_tokens("td") == 32

    native.error = True
    with pytest.raises(RuntimeError, match="injected failure"):
        proxy.allocate_slots(td, 1)
    assert get_resident_kv_tokens("td") == 32

    with pytest.raises(ValueError, match="Unsupported token-drop"):
        proxy.allocate_slots(td, 1, num_lookahead_tokens=1)
    assert get_resident_kv_tokens("td") == 32


def test_proxy_is_instance_local_and_never_replaces_allocator_class(proxy_factory):
    stock_method = _FakeManager.allocate_slots
    first = _FakeManager()
    second = _FakeManager()
    proxy = proxy_factory(first)
    td = _request("td", 104, token_drop=True)
    set_resident_kv_tokens("td", 32)

    proxy.allocate_slots(td, 1)
    assert _FakeManager.allocate_slots is stock_method
    assert proxy._manager is first
    assert second.calls == []
    normal = _request("normal", 104, token_drop=False)
    assert second.allocate_slots(normal, 1) is second.result
    assert second.calls[-1][0] is normal


def test_scheduler_configuration_is_opt_in(proxy_factory):
    from vllm.v1.core.sched.scheduler import Scheduler

    from lmcache.integration.vllm.experimental.token_drop_scheduler_proxy import (
        LMCacheTokenDropScheduler,
    )

    assert issubclass(LMCacheTokenDropScheduler, Scheduler)
