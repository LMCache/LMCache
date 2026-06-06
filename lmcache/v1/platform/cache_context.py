# SPDX-License-Identifier: Apache-2.0
"""Platform-agnostic cache-context factory.

The concrete implementations live in their respective sub-packages:

* :class:`~lmcache.v1.multiprocess.gpu_context.GPUCacheContext` --
  CUDA-backed.

:func:`create_cache_context` keeps the dispatch out of the call site
in :mod:`lmcache.v1.multiprocess.server` so adding a new accelerator
only requires shipping a new sub-package + extending the factory below.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import KVCache

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import LMCacheGroupView


def create_cache_context(
    kv_caches: KVCache,
    lmcache_logical_chunk_size: int = 256,
    layout_hints: LayoutHints | None = None,
    group_views: "Sequence[LMCacheGroupView]" = (),
    engine_type: EngineType = EngineType.VLLM,
) -> Any:
    """Create the appropriate cache context.

    The signature mirrors :class:`GPUCacheContext` so callers can
    forward their kwargs verbatim and stay agnostic of the active
    backend.

    Currently only :class:`GPUCacheContext` is supported.  CPU and
    other accelerator backends will be added in follow-up PRs.

    Args:
        kv_caches: KV cache tensor wrappers from the serving engine.
            Must be non-empty.
        lmcache_logical_chunk_size: Number of tokens per LMCache chunk.
        layout_hints: Optional hints for GPU KV format detection.
            Forwarded verbatim to the concrete context constructor.
        group_views: Engine-neutral KV cache group metadata.
        engine_type: Which serving engine produced the caches.

    Returns:
        A concrete cache context instance (currently always
        :class:`~lmcache.v1.multiprocess.gpu_context.GPUCacheContext`).

    Raises:
        ValueError: If *kv_caches* is empty.
    """
    # First Party
    from lmcache.v1.multiprocess.gpu_context import GPUCacheContext

    if not kv_caches:
        raise ValueError("create_cache_context requires a non-empty kv_caches list")

    return GPUCacheContext(
        kv_caches,
        lmcache_logical_chunk_size,
        layout_hints,
        group_views,
        engine_type,
    )
