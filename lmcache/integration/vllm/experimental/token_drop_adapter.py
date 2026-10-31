# SPDX-License-Identifier: Apache-2.0
"""vLLM integration adaptors for token dropping."""

# Future
from __future__ import annotations

# Standard
from functools import wraps
from typing import Any

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

    # Strip local version labels (+cpu, +cu130, ...) and keep the release only.
    if vllm_version.split("+", 1)[0] != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Token dropping requires vLLM {_SUPPORTED_VLLM_VERSION}, "
            f"got {vllm_version}"
        )


class _ResidentKVRequestView:
    """Let the allocator read resident KV length without changing the request."""

    def __init__(self, request, resident_kv_length):
        self._request = request
        self._resident_kv_length = resident_kv_length

    @property
    def num_computed_tokens(self):
        return self._resident_kv_length

    def __getattr__(self, name):
        """Read other attributes from the original request."""
        return getattr(self._request, name)


def _allocate_with_resident_frontier(
    original, manager, request, num_new_tokens, *args, **kwargs
):
    request_id = request.request_id
    resident_kv_length = get_resident_kv_tokens(request_id)
    if resident_kv_length is None:
        return original(manager, request, num_new_tokens, *args, **kwargs)
    if request.num_computed_tokens == 0 or "lmcache.token_drop" not in (
        extract_request_configs_from_request(request) or {}
    ):
        # A preempted request must recompute from the logical beginning.
        clear_resident_kv_tokens(request_id)
        return original(manager, request, num_new_tokens, *args, **kwargs)

    # These are vLLM 0.25.1 allocate_slots() options. Prefix/external KV,
    # speculative or encoder tokens, and full-sequence admission need
    # extra allocation handling that token dropping doesn't support in the MVP.
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

    new_blocks = original(
        manager,
        _ResidentKVRequestView(request, resident_kv_length),
        num_new_tokens,
        *args,
        **kwargs,
    )

    if new_blocks is not None:
        new_resident_kv_length = min(
            resident_kv_length + num_new_tokens, manager.max_model_len
        )
        set_resident_kv_tokens(request_id, new_resident_kv_length)
    return new_blocks


def install_token_drop_allocator_adapter() -> None:
    """Install the resident-frontier allocation hook for vLLM 0.25.1."""
    _check_vllm_version()

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_allocate_slots = KVCacheManager.allocate_slots
    adapter_marker = "_lmcache_token_drop_allocator_adapter"
    if getattr(original_allocate_slots, adapter_marker, False):
        return

    @wraps(original_allocate_slots)
    def wrapped_allocate_slots(self, *args, **kwargs):
        return _allocate_with_resident_frontier(
            original_allocate_slots, self, *args, **kwargs
        )

    # Prevent wrapping the allocator more than once.
    setattr(wrapped_allocate_slots, adapter_marker, True)
    KVCacheManager.allocate_slots = wrapped_allocate_slots


def install_token_drop_query_hooks(worker, no_compile_layers: dict[str, Any]) -> None:
    """Capture post-RoPE queries without changing attention forward outputs."""
    _check_vllm_version()
    if worker._original_attention_forwards:
        return

    original_forwards = {}
    for layer_name in worker._kv_caches:
        impl = no_compile_layers[layer_name].impl
        original_forwards[layer_name] = (impl, impl.forward)

    worker._original_attention_forwards = original_forwards
    for impl, original_forward in original_forwards.values():

        def forward_with_query_capture(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            *args,
            _forward=original_forward,
            **kwargs,
        ):
            # Profiling forwards have no attention metadata or live queries.
            if attn_metadata is not None:
                # PIECEWISE CUDA graphs may pad the query tensor.
                worker.capture_query(
                    attn_layer.layer_name, query[: attn_metadata.num_actual_tokens]
                )
            return _forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                *args,
                **kwargs,
            )

        try:
            setattr(impl, "forward", forward_with_query_capture)  # noqa: B010
        except Exception:
            remove_token_drop_query_hooks(worker)
            raise


def remove_token_drop_query_hooks(worker) -> None:
    """Restore original attention forwards after query capture."""
    for impl, original_forward in worker._original_attention_forwards.values():
        setattr(impl, "forward", original_forward)  # noqa: B010
    worker._original_attention_forwards.clear()
