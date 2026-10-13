# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.token_drop_allocator_adapter import (
    _allocate_with_resident_frontier,
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


@pytest.fixture(autouse=True)
def clear_state():
    for req in ("td", "normal"):
        clear_resident_kv_tokens(req)
    yield
    for req in ("td", "normal"):
        clear_resident_kv_tokens(req)


def request(request_id="td", *, td=True, logical=100):
    return SimpleNamespace(
        request_id=request_id,
        num_computed_tokens=logical,
        kv_transfer_params=(
            {"lmcache.token_drop": {"algorithm": "rkv", "config": {}}} if td else None
        ),
    )


def allocate(original, req, new_tokens=1):
    return _allocate_with_resident_frontier(
        original,
        SimpleNamespace(max_model_len=1024),
        req,
        new_tokens,
    )


def test_resident_frontier_only_for_token_drop():
    req = request()
    set_resident_kv_tokens("td", 16)
    seen = []

    def original(manager, request, *args):
        seen.append(request.num_computed_tokens)
        return object()

    assert allocate(original, req) is not None
    assert seen == [16]
    assert req.num_computed_tokens == 100
    assert get_resident_kv_tokens("td") == 17


def test_normal_request_passes_through_and_clears_stale_state():
    req = request("normal", td=False)
    set_resident_kv_tokens("normal", 16)
    seen = []

    def original(manager, request, *args):
        seen.append(request.num_computed_tokens)
        return object()

    assert allocate(original, req) is not None
    assert seen == [100]
    assert get_resident_kv_tokens("normal") is None


def test_failed_allocation_keeps_resident_frontier():
    req = request()
    set_resident_kv_tokens("td", 16)
    assert allocate(lambda manager, request, *args: None, req) is None
    assert req.num_computed_tokens == 100
    assert get_resident_kv_tokens("td") == 16


def test_preempted_request_returns_to_vanilla_allocation():
    req = request(logical=0)
    set_resident_kv_tokens("td", 16)
    seen = []

    def original(manager, request, *args):
        seen.append(request.num_computed_tokens)
        return object()

    allocate(original, req)
    assert seen == [0]
    assert get_resident_kv_tokens("td") is None


def test_rejects_unsupported_resident_allocation_mode():
    req = request()
    set_resident_kv_tokens("td", 16)
    with pytest.raises(ValueError, match="only supports normal synchronous"):
        _allocate_with_resident_frontier(
            lambda *args: object(),
            SimpleNamespace(max_model_len=1024),
            req,
            1,
            num_lookahead_tokens=1,
        )
    assert req.num_computed_tokens == 100
    assert get_resident_kv_tokens("td") == 16
