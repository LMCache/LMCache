# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM 0.25.1 token-dropping adaptor (currently allocator only)."""

# Future
from __future__ import annotations

# Standard
from functools import wraps

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)
from lmcache.integration.vllm.utils import extract_request_configs_from_request

_SUPPORTED_VLLM_VERSION = "0.25.1"


def _check_vllm_version() -> None:
    # Third Party
    from vllm.version import __version__ as vllm_version

    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Token dropping requires vLLM {_SUPPORTED_VLLM_VERSION}, "
            f"got {vllm_version}"
        )


class _PhysicalKVRequestView:
    """Let the allocator read resident KV length without changing the request."""

    def __init__(self, request, physical_kv_length):
        self._request = request
        self.physical_kv_length = physical_kv_length

    @property
    def num_computed_tokens(self):
        return self.physical_kv_length

    def __getattr__(self, name):
        return getattr(self._request, name)


def _allocate_with_resident_frontier(
    original, manager, request, num_new_tokens, *args, **kwargs
):
    request_id = request.request_id
    resident = get_resident_kv_tokens(request_id)
    if resident is None:
        return original(manager, request, num_new_tokens, *args, **kwargs)

    resident_kv_length: int = resident
    if request.num_computed_tokens == 0 or "lmcache.token_drop" not in (
        extract_request_configs_from_request(request) or {}
    ):
        # A preempted request must recompute from the logical beginning.
        clear_resident_kv_tokens(request_id)
        return original(manager, request, num_new_tokens, *args, **kwargs)

    # These are vLLM 0.25.1 allocate_slots() options. Prefix/external KV,
    # speculative or encoder tokens, and full-sequence admission need
    # allocation accounting not supported by compacted KV requests.
    if (
        args
        or kwargs.get("new_computed_blocks") is not None
        or any(
            kwargs.get(name)
            for name in (
                "num_new_computed_tokens",
                "num_external_computed_tokens",
                "num_lookahead_tokens",
                "num_encoder_tokens",
                "full_sequence_must_fit",
            )
        )
    ):
        raise ValueError("Unsupported token-drop physical allocation mode")

    result = original(
        manager,
        _PhysicalKVRequestView(request, resident_kv_length),
        num_new_tokens,
        *args,
        **kwargs,
    )

    if result is not None:
        updated_length = min(resident_kv_length + num_new_tokens, manager.max_model_len)
        set_resident_kv_tokens(request_id, updated_length)
    return result


def install_token_drop_allocator_adapter() -> None:
    """Install the resident-frontier allocation hook for vLLM 0.25.1."""
    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _check_vllm_version()

    original_allocate = KVCacheManager.allocate_slots
    if getattr(original_allocate, "_lmcache_token_drop_allocator_adapter", False):
        return

    @wraps(original_allocate)
    def wrapped_allocate(self, *args, **kwargs):
        return _allocate_with_resident_frontier(
            original_allocate, self, *args, **kwargs
        )

    # Mark the wrapper so repeated installation does not stack allocator hooks.
    wrapped_allocate._lmcache_token_drop_allocator_adapter = True  # type: ignore[attr-defined]
    KVCacheManager.allocate_slots = wrapped_allocate
