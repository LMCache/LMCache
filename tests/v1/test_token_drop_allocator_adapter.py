# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from lmcache.integration.vllm.token_drop_allocator_adapter import (
    _allocate_with_resident_frontier,
    _keep_token_drop_blocks_private,
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    install_token_drop_allocator_adapter,
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
    manager.seen_delay_cache_blocks = delay_cache_blocks
    return manager.result


def _request(logical: int, *, token_drop: bool = True):
    transfer = (
        {
            "lmcache.token_drop": {
                "algorithm": "fake",
                "config": {},
            }
        }
        if token_drop
        else {}
    )
    return SimpleNamespace(
        request_id="req",
        num_computed_tokens=logical,
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": transfer}
        ),
    )


@pytest.fixture(autouse=True)
def _clear_resident_state():
    clear_resident_kv_tokens("req")
    yield
    clear_resident_kv_tokens("req")


def test_resident_frontier_is_used_only_for_token_drop_allocation():
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


def test_normal_request_is_exact_vanilla_passthrough():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104, token_drop=False)
    set_resident_kv_tokens(request.request_id, 32)

    result = _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        1,
        delay_cache_blocks=True,
    )

    assert result is manager.result
    assert manager.seen_num_computed_tokens == 104
    assert manager.seen_delay_cache_blocks is True
    assert get_resident_kv_tokens(request.request_id) is None


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


def test_preemption_returns_to_vanilla_frontier_without_changing_cache_flag():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(0)
    set_resident_kv_tokens(request.request_id, 32)

    _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        8,
        delay_cache_blocks=False,
    )

    assert manager.seen_num_computed_tokens == 0
    assert manager.seen_delay_cache_blocks is False
    assert get_resident_kv_tokens(request.request_id) is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_new_computed_tokens", 1),
        ("new_computed_blocks", object()),
        ("num_lookahead_tokens", 1),
        ("num_external_computed_tokens", 1),
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


def test_token_drop_block_promotion_is_skipped_but_normal_request_is_unchanged():
    calls = []

    def original(
        block_pool,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask=None,
    ):
        calls.append(
            (
                block_pool,
                request.request_id,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
                block_mask,
            )
        )
        return "cached"

    block_pool = object()
    blocks = [object()]
    td_request = _request(32)
    normal_request = _request(32, token_drop=False)

    assert (
        _keep_token_drop_blocks_private(
            original,
            block_pool,
            td_request,
            blocks,
            0,
            2,
            16,
            0,
            None,
        )
        is None
    )
    assert calls == []

    assert (
        _keep_token_drop_blocks_private(
            original,
            block_pool,
            normal_request,
            blocks,
            0,
            2,
            16,
            0,
            None,
        )
        == "cached"
    )
    assert calls == [
        (block_pool, "req", blocks, 0, 2, 16, 0, None)
    ]


def test_private_blocks_keep_vllm_running_request_bookkeeping() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    calls = []
    manager = SimpleNamespace(
        num_cached_block={},
        block_size=16,
        scheduler_block_size=16,
        kv_cache_spec=object(),
        use_eagle=False,
        kv_cache_group_id=0,
        req_to_blocks={"req": [object(), object()]},
        block_pool=SimpleNamespace(
            cache_full_blocks=lambda **kwargs: calls.append(kwargs)
        ),
        reachable_block_mask=lambda **_kwargs: None,
    )
    request = SimpleNamespace(request_id="req", num_prompt_tokens=16)

    SingleTypeKVCacheManager.cache_blocks(manager, request, 32)

    assert manager.num_cached_block["req"] == 2
    assert len(calls) == 1


def test_install_allocator_adaptor_is_idempotent_for_pinned_vllm():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_allocate = KVCacheManager.allocate_slots
    original_cache = BlockPool.cache_full_blocks
    try:
        install_token_drop_allocator_adapter()
        wrapped_allocate = KVCacheManager.allocate_slots
        wrapped_cache = BlockPool.cache_full_blocks

        assert getattr(
            wrapped_allocate,
            "_lmcache_token_drop_allocator_adapter",
            False,
        )
        assert getattr(
            wrapped_cache,
            "_lmcache_token_drop_private_blocks",
            False,
        )

        install_token_drop_allocator_adapter()
        assert KVCacheManager.allocate_slots is wrapped_allocate
        assert BlockPool.cache_full_blocks is wrapped_cache
    finally:
        KVCacheManager.allocate_slots = original_allocate
        BlockPool.cache_full_blocks = original_cache


def test_resident_state_helpers():
    assert get_resident_kv_tokens("req") is None
    set_resident_kv_tokens("req", 7)
    assert get_resident_kv_tokens("req") == 7
    clear_resident_kv_tokens("req")
    assert get_resident_kv_tokens("req") is None
