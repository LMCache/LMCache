# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM runtime adapter for LMCache-owned resident KV allocation."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable

from lmcache.integration.vllm.token_drop import token_drop_spec_from_request

_SUPPORTED_VLLM_VERSION = "0.25.1"
_EXPECTED_ALLOCATE_PARAMS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_computed_tokens",
    "new_computed_blocks",
    "num_lookahead_tokens",
    "num_external_computed_tokens",
    "delay_cache_blocks",
    "num_encoder_tokens",
    "full_sequence_must_fit",
    "reserved_blocks",
    "has_scheduled_reqs",
)
_EXPECTED_PRIVATE_CACHE_PARAMS = (
    "self",
    "request",
    "blocks",
    "num_cached_blocks",
    "num_full_blocks",
    "block_size",
    "kv_cache_group_id",
    "block_mask",
)
_resident_kv_tokens: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens.pop(request_id, None)


def _allocate_with_resident_frontier(
    original: Callable[..., Any],
    manager: Any,
    request: Any,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: Any = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> Any:
    request_id = request.request_id
    token_drop_spec = token_drop_spec_from_request(request)
    resident = get_resident_kv_tokens(request_id)

    if token_drop_spec is None:
        # Request config is authoritative. Never let stale resident state turn
        # a normal request into token-dropping behavior.
        if resident is not None:
            clear_resident_kv_tokens(request_id)
        return original(
            manager,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )

    # vLLM preemption frees request blocks and resets logical progress to zero.
    # The next allocation returns to vanilla recompute semantics. APC privacy
    # is enforced separately at the block-promotion boundary.
    if request.num_computed_tokens == 0:
        clear_resident_kv_tokens(request_id)
        return original(
            manager,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )

    if resident is None:
        return original(
            manager,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )

    if (
        num_new_computed_tokens
        or new_computed_blocks is not None
        or num_lookahead_tokens
        or num_external_computed_tokens
        or num_encoder_tokens
        or full_sequence_must_fit
    ):
        raise ValueError(
            "Token-drop resident allocation only supports normal synchronous "
            "full-attention decoding"
        )

    logical_num_computed_tokens = request.num_computed_tokens
    request.num_computed_tokens = resident
    try:
        result = original(
            manager,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )
    finally:
        request.num_computed_tokens = logical_num_computed_tokens

    if result is not None:
        set_resident_kv_tokens(
            request_id,
            min(resident + num_new_tokens, manager.max_model_len),
        )
    return result


def _keep_token_drop_blocks_private(
    original: Callable[..., Any],
    block_pool: Any,
    request: Any,
    blocks: Any,
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
    block_mask: Any = None,
) -> Any:
    """Skip only APC hash/index publication for token-drop blocks."""
    if token_drop_spec_from_request(request) is not None:
        return None
    return original(
        block_pool,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask,
    )


def install_token_drop_allocator_adapter() -> None:
    """Install resident-frontier and private-block adaptors for vLLM 0.25.1."""
    # Third Party
    from vllm.version import __version__ as vllm_version
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "Token dropping requires vLLM "
            f"{_SUPPORTED_VLLM_VERSION}, got {vllm_version}"
        )

    original_allocate = KVCacheManager.allocate_slots
    if not getattr(
        original_allocate,
        "_lmcache_token_drop_allocator_adapter",
        False,
    ):
        params = tuple(signature(original_allocate).parameters)
        if params != _EXPECTED_ALLOCATE_PARAMS:
            raise RuntimeError(
                "Unsupported vLLM KVCacheManager.allocate_slots signature: "
                f"{params}"
            )

        @wraps(original_allocate)
        def wrapped_allocate(
            self: Any,
            request: Any,
            num_new_tokens: int,
            num_new_computed_tokens: int = 0,
            new_computed_blocks: Any = None,
            num_lookahead_tokens: int = 0,
            num_external_computed_tokens: int = 0,
            delay_cache_blocks: bool = False,
            num_encoder_tokens: int = 0,
            full_sequence_must_fit: bool = False,
            reserved_blocks: int = 0,
            has_scheduled_reqs: bool = True,
        ) -> Any:
            return _allocate_with_resident_frontier(
                original_allocate,
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens,
                num_external_computed_tokens,
                delay_cache_blocks,
                num_encoder_tokens,
                full_sequence_must_fit,
                reserved_blocks,
                has_scheduled_reqs,
            )

        wrapped_allocate._lmcache_token_drop_allocator_adapter = True  # type: ignore[attr-defined]
        KVCacheManager.allocate_slots = wrapped_allocate

    # This is the actual unhashed->cached promotion boundary. Keep the
    # manager-level cache_blocks() call intact so vLLM still updates its normal
    # running-request bookkeeping (num_cached_block); only hash/index publication
    # is skipped for token-drop requests.
    original_cache = BlockPool.cache_full_blocks
    if not getattr(original_cache, "_lmcache_token_drop_private_blocks", False):
        params = tuple(signature(original_cache).parameters)
        if params != _EXPECTED_PRIVATE_CACHE_PARAMS:
            raise RuntimeError(
                f"Unsupported vLLM BlockPool.cache_full_blocks signature: {params}"
            )

        @wraps(original_cache)
        def wrapped_cache(
            self: Any,
            request: Any,
            blocks: Any,
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            kv_cache_group_id: int,
            block_mask: Any = None,
        ) -> Any:
            return _keep_token_drop_blocks_private(
                original_cache,
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
                block_mask,
            )

        wrapped_cache._lmcache_token_drop_private_blocks = True  # type: ignore[attr-defined]
        BlockPool.cache_full_blocks = wrapped_cache
