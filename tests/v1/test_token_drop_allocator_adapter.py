# SPDX-License-Identifier: Apache-2.0
"""Tests for token-dropping KV allocation."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import sys

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _allocate_with_resident_frontier,
    install_token_drop_allocator_adapter,
)
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    ResidentKVState,
    bind_resident_kv_state,
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
    manager.seen_computed_tokens = request.num_computed_tokens
    manager.seen_request = request
    manager.seen_kwargs = kwargs
    return manager.new_blocks


class _Pool:
    pass


@pytest.fixture
def resident_state():
    return ResidentKVState()


def _manager(resident_state, **kwargs):
    pool = _Pool()
    bind_resident_kv_state(pool, resident_state)
    return SimpleNamespace(block_pool=pool, max_model_len=4096, **kwargs)


def test_td_allocation_uses_resident_length_without_changing_logical_length(
    resident_state,
):
    new_blocks = object()
    manager = _manager(resident_state, new_blocks=new_blocks)
    request = _request(104)
    resident_state.set("req", 32)

    assert (
        _allocate_with_resident_frontier(_allocate, manager, request, 1) is new_blocks
    )
    assert manager.seen_computed_tokens == 32
    assert manager.seen_request is not request
    assert manager.seen_request.request_id == request.request_id
    assert request.num_computed_tokens == 104
    assert resident_state.get("req") == 33


def test_without_resident_length_uses_original_request(resident_state):
    new_blocks = object()
    manager = _manager(resident_state, new_blocks=new_blocks)
    request = _request(0)

    assert (
        _allocate_with_resident_frontier(_allocate, manager, request, 8) is new_blocks
    )
    assert manager.seen_computed_tokens == 0
    assert manager.seen_request is request
    assert resident_state.get("req") is None


def test_normal_request_ignores_stale_resident_length(resident_state):
    new_blocks = object()
    manager = _manager(resident_state, new_blocks=new_blocks)
    request = _request(104, token_drop=False)
    resident_state.set("req", 32)

    assert (
        _allocate_with_resident_frontier(
            _allocate, manager, request, 1, delay_cache_blocks=True
        )
        is new_blocks
    )
    assert manager.seen_computed_tokens == 104
    assert manager.seen_request is request
    assert manager.seen_kwargs == {"delay_cache_blocks": True}
    assert resident_state.get("req") is None


def test_failed_allocation_keeps_resident_length(resident_state):
    manager = _manager(resident_state, new_blocks=None)
    request = _request(104)
    resident_state.set("req", 32)

    assert _allocate_with_resident_frontier(_allocate, manager, request, 1) is None
    assert resident_state.get("req") == 32
    assert request.num_computed_tokens == 104


def test_allocation_exception_never_changes_original_logical_request(resident_state):
    manager = _manager(resident_state)
    request = _request(104)
    resident_state.set("req", 32)

    def fail_allocation(_manager, allocation_request, _num_new_tokens):
        assert allocation_request.num_computed_tokens == 32
        assert request.num_computed_tokens == 104
        raise RuntimeError("injected allocation error")

    with pytest.raises(RuntimeError, match="injected allocation error"):
        _allocate_with_resident_frontier(fail_allocation, manager, request, 1)

    assert request.num_computed_tokens == 104
    assert resident_state.get("req") == 32


def test_zero_logical_progress_clears_stale_resident_length(resident_state):
    manager = _manager(resident_state, new_blocks=object())
    request = _request(0)
    resident_state.set("req", 32)

    _allocate_with_resident_frontier(_allocate, manager, request, 8)

    assert manager.seen_computed_tokens == 0
    assert manager.seen_request is request
    assert resident_state.get("req") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"num_new_computed_tokens": 1}, id="cached-prefix-hit"),
        pytest.param({"num_external_computed_tokens": 1}, id="external-kv-restore"),
        pytest.param({"num_lookahead_tokens": 1}, id="speculative-lookahead"),
        pytest.param({"new_computed_blocks": object()}, id="cached-blocks"),
        pytest.param({"num_encoder_tokens": 1}, id="encoder-tokens"),
        pytest.param({"full_sequence_must_fit": True}, id="full-sequence-admission"),
    ],
)
def test_td_physical_allocation_rejects_incompatible_kv_modes(kwargs, resident_state):
    manager = _manager(resident_state, new_blocks=object())
    request = _request(104)
    resident_state.set("req", 32)

    original_allocate = Mock()
    with pytest.raises(ValueError, match="Unsupported token-drop physical"):
        _allocate_with_resident_frontier(
            original_allocate, manager, request, 1, **kwargs
        )

    original_allocate.assert_not_called()
    assert request.num_computed_tokens == 104
    assert resident_state.get("req") == 32


def test_allocator_hook_rejects_unsupported_vllm_version():
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    version = ModuleType("vllm.version")
    version.__version__ = "0.26.0"

    with patch.dict(sys.modules, {"vllm": vllm, "vllm.version": version}):
        with pytest.raises(RuntimeError, match="requires vLLM 0.25.1, got 0.26.0"):
            install_token_drop_allocator_adapter()


def test_allocator_hook_installation_is_idempotent():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def original_allocate_slots(self, request, num_new_tokens):
        return None

    with patch.object(KVCacheManager, "allocate_slots", original_allocate_slots):
        install_token_drop_allocator_adapter()
        wrapped = KVCacheManager.allocate_slots
        assert wrapped is not original_allocate_slots

        install_token_drop_allocator_adapter()
        assert KVCacheManager.allocate_slots is wrapped


def test_allocator_hook_isolates_native_pools_with_identical_request_ids():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def make_manager(state):
        pool = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16)
        if state is not None:
            bind_resident_kv_state(pool, state)
        return SimpleNamespace(block_pool=pool, max_model_len=4096, new_blocks=object())

    first, second = ResidentKVState(), ResidentKVState()
    first.set("req", 16)
    second.set("req", 32)
    a, b, ordinary = make_manager(first), make_manager(second), make_manager(None)
    with patch.object(KVCacheManager, "allocate_slots", _allocate):
        install_token_drop_allocator_adapter()
        for manager, expected in ((a, 16), (b, 32), (ordinary, 104)):
            request = _request(104)
            assert (
                KVCacheManager.allocate_slots(manager, request, 1) is manager.new_blocks
            )
            assert manager.seen_computed_tokens == expected
            assert request.num_computed_tokens == 104
        assert first.get("req") == 17
        assert second.get("req") == 33
        KVCacheManager.allocate_slots(a, _request(0), 1)
        assert first.get("req") is None
        assert second.get("req") == 33
