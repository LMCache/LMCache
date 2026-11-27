# SPDX-License-Identifier: Apache-2.0
"""vLLM integration adaptors for token dropping."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, cast

# Third Party
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)
from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    # Third Party
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backend import AttentionImpl
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


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


def install_token_drop_prefix_caching_publication_guard() -> None:
    """Prevent token-dropping requests from publishing KV blocks to the prefix cache.

    In-place KV compaction requires private blocks from initial allocation.
    Prefix-cache lookup must be disabled separately before enabling token dropping.
    """
    _check_vllm_version()

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # Wrap BlockPool.cache_full_blocks() to skip publication for
    # token-dropping requests.
    original_cache_full_blocks = BlockPool.cache_full_blocks
    guard_marker = "_lmcache_token_drop_prefix_caching_publication_guard"
    if getattr(original_cache_full_blocks, guard_marker, False):
        return

    @wraps(original_cache_full_blocks)
    def cache_full_blocks_with_publication_guard(self, request, *args, **kwargs):
        if "lmcache.token_drop" in (
            extract_request_configs_from_request(request) or {}
        ):
            return
        return original_cache_full_blocks(self, request, *args, **kwargs)

    # Prevent wrapping the prefix-cache publication method more than once.
    setattr(cache_full_blocks_with_publication_guard, guard_marker, True)
    BlockPool.cache_full_blocks = cache_full_blocks_with_publication_guard


class TokenDropQueryHooks:
    """Capture post-RoPE Q with reversible attention hooks."""

    def __init__(self) -> None:
        """Store per-layer state for restoring attention forwards."""
        self._restore_state: dict[
            str,
            tuple[
                AttentionImpl[FlashAttentionMetadata], Callable[..., torch.Tensor], bool
            ],
        ] = {}

    def install(
        self,
        layer_names: Iterable[str],
        no_compile_layers: Mapping[str, Attention],
        capture_query: Callable[[str, torch.Tensor], None],
    ) -> None:
        """Install query capture hooks on attention layers.

        Requires FlashAttention with FULL CUDA Graph replay disabled.
        Repeated installs are no-ops. Failed installs roll back.

        Args:
            layer_names: Layers to hook.
            no_compile_layers: vLLM mapping used to look up attention layers by name.
            capture_query: Callback receiving the layer name and a borrowed,
                read-only view of the post-RoPE Q tensor.

        Raises:
            RuntimeError: Unsupported vLLM version.
            KeyError: Missing layer.
        """
        _check_vllm_version()
        if self._restore_state:
            return

        restore_state = {}
        for layer_name in layer_names:
            attention_impl = cast(
                "AttentionImpl[FlashAttentionMetadata]",
                no_compile_layers[layer_name].impl,
            )
            restore_state[layer_name] = (
                attention_impl,
                attention_impl.forward,
                "forward" in vars(attention_impl),
            )

        self._restore_state = restore_state
        for attention_impl, original_forward, _ in restore_state.values():

            def forward_with_query_capture(
                attn_layer: Attention,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata: FlashAttentionMetadata | None,
                *args: object,
                _original_forward: Callable[..., torch.Tensor] = original_forward,
                **kwargs: object,
            ) -> torch.Tensor:
                if attn_metadata is not None:
                    capture_query(
                        attn_layer.layer_name, query[: attn_metadata.num_actual_tokens]
                    )
                return _original_forward(
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
                attention_impl.forward = forward_with_query_capture
            except Exception:
                # Roll back earlier hooks if a later layer rejects installation.
                self.remove()
                raise

    def remove(self) -> None:
        """Remove hooks and restore the original forwards."""
        for (
            attention_impl,
            original_forward,
            had_instance_forward_override,
        ) in self._restore_state.values():
            if had_instance_forward_override:
                attention_impl.forward = original_forward
            elif "forward" in vars(attention_impl):
                # Remove the instance override to restore the original class method.
                delattr(attention_impl, "forward")
        self._restore_state.clear()
