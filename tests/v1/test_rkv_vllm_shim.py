# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from lmcache.integration.vllm.rkv_vllm_shim import (
    _allocate_with_resident_frontier,
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


def _original_allocate(
    manager,
    request,
    num_new_tokens,
    num_new_computed_tokens=0,
    new_computed_blocks=None,
    num_lookahead_tokens=0,
    num_external_computed_tokens=0,
    delay_cache_blocks=False,
    num_encoder_tokens=0,
    full_sequence_must_fit=False,
    reserved_blocks=0,
    has_scheduled_reqs=True,
):
    manager.seen_num_computed_tokens = request.num_computed_tokens
    manager.seen_num_new_tokens = num_new_tokens
    return manager.result


def _request(logical: int):
    return SimpleNamespace(request_id="req", num_computed_tokens=logical)


@pytest.fixture(autouse=True)
def _clear_resident_state():
    clear_resident_kv_tokens("req")
    yield
    clear_resident_kv_tokens("req")


def test_resident_frontier_is_used_only_during_allocation():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    result = _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        1,
    )

    assert result is manager.result
    assert manager.seen_num_computed_tokens == 32
    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 33


def test_failed_allocation_does_not_advance_resident_frontier():
    manager = SimpleNamespace(max_model_len=4096, result=None)
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    assert (
        _allocate_with_resident_frontier(
            _original_allocate,
            manager,
            request,
            1,
        )
        is None
    )

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 32


def test_preemption_returns_to_vanilla_allocation():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(0)
    set_resident_kv_tokens(request.request_id, 32)

    _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        8,
    )

    assert manager.seen_num_computed_tokens == 0
    assert get_resident_kv_tokens(request.request_id) is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_new_computed_tokens", 1),
        ("new_computed_blocks", object()),
        ("num_lookahead_tokens", 1),
        ("num_external_computed_tokens", 1),
        ("delay_cache_blocks", True),
        ("num_encoder_tokens", 1),
        ("full_sequence_must_fit", True),
    ],
)
def test_resident_frontier_rejects_unsupported_allocation_modes(name, value):
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    with pytest.raises(ValueError, match="normal synchronous full-attention"):
        _allocate_with_resident_frontier(
            _original_allocate,
            manager,
            request,
            1,
            **{name: value},
        )

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 32


def test_resident_state_helpers():
    assert get_resident_kv_tokens("req") is None
    set_resident_kv_tokens("req", 7)
    assert get_resident_kv_tokens("req") == 7
    clear_resident_kv_tokens("req")
    assert get_resident_kv_tokens("req") is None
