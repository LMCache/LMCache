# SPDX-License-Identifier: Apache-2.0
"""Prevent token-dropping blocks from entering vLLM's prefix cache."""

# Future
from __future__ import annotations

# Standard
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Callable

# First Party
from lmcache.integration.vllm.experimental.token_drop import parse_token_drop_spec
from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

_EXPECTED_PARAMS = (
    "self",
    "request",
    "blocks",
    "num_cached_blocks",
    "num_full_blocks",
    "block_size",
    "kv_cache_group_id",
    "block_mask",
)


def _cache_private_or_normal(
    original: Callable[..., None],
    pool: BlockPool,
    request: Request,
    blocks: list[KVCacheBlock],
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
    block_mask: list[bool] | None = None,
) -> None:
    # Preserve manager-level cache bookkeeping; skip only hash/index publication.
    if parse_token_drop_spec(extract_request_configs_from_request(request)) is not None:
        return
    return original(
        pool,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask,
    )


def install_token_drop_apc_guard() -> None:
    """Install the opt-in APC publication guard for pinned vLLM 0.25.1."""
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.version import __version__ as vllm_version

    if vllm_version.split("+", 1)[0] != "0.25.1":
        raise RuntimeError(f"Token dropping requires vLLM 0.25.1, got {vllm_version}")

    original_get = KVCacheManager.get_computed_blocks
    if not getattr(original_get, "_lmcache_td_guard", False):

        @wraps(original_get)
        def guarded_get(self, request):
            if (
                parse_token_drop_spec(extract_request_configs_from_request(request))
                is not None
            ):
                return self.empty_kv_cache_blocks, 0
            return original_get(self, request)

        guarded_get._lmcache_td_guard = True  # type: ignore[attr-defined]
        KVCacheManager.get_computed_blocks = guarded_get

    original = BlockPool.cache_full_blocks
    if getattr(original, "_lmcache_token_drop_apc_guard", False):
        return
    params = tuple(signature(original).parameters)
    if params != _EXPECTED_PARAMS:
        raise RuntimeError(
            f"Unsupported BlockPool.cache_full_blocks signature: {params}"
        )

    @wraps(original)
    def wrapped(
        self: BlockPool,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        return _cache_private_or_normal(
            original,
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask,
        )

    wrapped._lmcache_token_drop_apc_guard = True  # type: ignore[attr-defined]
    BlockPool.cache_full_blocks = wrapped
