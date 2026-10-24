# SPDX-License-Identifier: Apache-2.0
"""Essential resident-frontier allocation correctness tests."""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _allocate_with_resident_frontier,
    install_token_drop_allocator_adapter,
)
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


def _request(logical_tokens, *, token_drop=True):
    configs = (
        {"lmcache.token_drop": {"algorithm": "rkv", "config": {}}} if token_drop else {}
    )
    return SimpleNamespace(
        request_id="req",
        num_computed_tokens=logical_tokens,
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": configs}),
    )


def _allocate(manager, request, num_new_tokens, *args, **kwargs):
    manager.seen_logical_tokens = request.num_computed_tokens
    manager.seen_request = request
    manager.seen_args = args
    manager.seen_kwargs = kwargs
    if manager.error:
        raise RuntimeError("injected allocation error")
    return manager.result


@pytest.fixture(autouse=True)
def _reset_resident_frontier():
    clear_resident_kv_tokens("req")
    yield
    clear_resident_kv_tokens("req")


def test_td_allocation_uses_separate_physical_view_without_changing_logical_request():
    result = object()
    manager = SimpleNamespace(max_model_len=4096, result=result, error=False)
    request = _request(104)
    set_resident_kv_tokens("req", 32)

    assert _allocate_with_resident_frontier(_allocate, manager, request, 1) is result
    assert manager.seen_logical_tokens == 32
    assert manager.seen_request is not request
    assert manager.seen_request.physical_kv_length == 32
    assert manager.seen_request.request_id == request.request_id
    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens("req") == 33


def test_td_first_prefill_without_physical_frontier_uses_vllm_allocation():
    result = object()
    manager = SimpleNamespace(max_model_len=4096, result=result, error=False)
    request = _request(0)

    assert _allocate_with_resident_frontier(_allocate, manager, request, 8) is result
    assert manager.seen_logical_tokens == 0
    assert get_resident_kv_tokens("req") is None


def test_normal_allocation_passes_through_without_using_physical_frontier():
    result = object()
    manager = SimpleNamespace(max_model_len=4096, result=result, error=False)
    request = _request(104, token_drop=False)
    set_resident_kv_tokens("req", 32)

    assert (
        _allocate_with_resident_frontier(
            _allocate, manager, request, 1, delay_cache_blocks=True
        )
        is result
    )
    assert manager.seen_logical_tokens == 104
    assert manager.seen_request is request
    assert manager.seen_kwargs == {"delay_cache_blocks": True}
    assert get_resident_kv_tokens("req") is None


def test_failed_allocation_does_not_advance_physical_frontier():
    manager = SimpleNamespace(max_model_len=4096, result=None, error=False)
    request = _request(104)
    set_resident_kv_tokens("req", 32)

    assert _allocate_with_resident_frontier(_allocate, manager, request, 1) is None
    assert get_resident_kv_tokens("req") == 32
    assert request.num_computed_tokens == 104


def test_allocation_exception_never_changes_original_logical_request():
    manager = SimpleNamespace(max_model_len=4096)
    request = _request(104)
    set_resident_kv_tokens("req", 32)

    def fail_allocation(_manager, allocation_request, _num_new_tokens):
        assert allocation_request.num_computed_tokens == 32
        assert request.num_computed_tokens == 104
        raise RuntimeError("injected allocation error")

    with pytest.raises(RuntimeError, match="injected allocation error"):
        _allocate_with_resident_frontier(fail_allocation, manager, request, 1)

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens("req") == 32


def test_preempted_td_request_recomputes_from_logical_start():
    manager = SimpleNamespace(max_model_len=4096, result=object(), error=False)
    request = _request(0)
    set_resident_kv_tokens("req", 32)

    _allocate_with_resident_frontier(_allocate, manager, request, 8)

    assert manager.seen_logical_tokens == 0
    assert get_resident_kv_tokens("req") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"num_new_computed_tokens": 1}, id="cached-prefix-hit"),
        pytest.param({"num_external_computed_tokens": 1}, id="external-kv-restore"),
        pytest.param({"num_lookahead_tokens": 1}, id="speculative-lookahead"),
    ],
)
def test_td_physical_allocation_rejects_incompatible_kv_modes(kwargs):
    manager = SimpleNamespace(max_model_len=4096, result=object(), error=False)
    request = _request(104)
    set_resident_kv_tokens("req", 32)

    with pytest.raises(ValueError, match="Unsupported token-drop physical"):
        _allocate_with_resident_frontier(_allocate, manager, request, 1, **kwargs)

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens("req") == 32


def test_allocator_hook_installation_is_idempotent():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__ != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original = KVCacheManager.allocate_slots
    with patch.object(KVCacheManager, "allocate_slots", original):
        install_token_drop_allocator_adapter()
        installed = KVCacheManager.allocate_slots
        assert getattr(installed, "_lmcache_token_drop_allocator_adapter", False)

        install_token_drop_allocator_adapter()
        assert KVCacheManager.allocate_slots is installed
