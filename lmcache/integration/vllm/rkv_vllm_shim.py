# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM runtime shim for LMCache-owned resident KV allocation."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable

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
    resident = get_resident_kv_tokens(request_id)
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

    # vLLM preemption frees request blocks and resets logical progress to zero.
    # The next allocation must therefore return to vanilla recompute semantics.
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

    if (
        num_new_computed_tokens
        or new_computed_blocks is not None
        or num_lookahead_tokens
        or num_external_computed_tokens
        or delay_cache_blocks
        or num_encoder_tokens
        or full_sequence_must_fit
    ):
        raise ValueError(
            "R-KV resident allocation only supports normal synchronous "
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


def install_rkv_vllm_allocator_shim() -> None:
    """Install the resident-KV allocation wrapper for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.version import __version__ as vllm_version
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"R-KV MVP requires vLLM {_SUPPORTED_VLLM_VERSION}, got {vllm_version}"
        )

    original = KVCacheManager.allocate_slots
    if getattr(original, "_lmcache_rkv_allocator_shim", False):
        return

    params = tuple(signature(original).parameters)
    if params != _EXPECTED_ALLOCATE_PARAMS:
        raise RuntimeError(
            f"Unsupported vLLM KVCacheManager.allocate_slots signature: {params}"
        )

    @wraps(original)
    def wrapped(
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
            original,
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

    wrapped._lmcache_rkv_allocator_shim = True  # type: ignore[attr-defined]
    KVCacheManager.allocate_slots = wrapped
