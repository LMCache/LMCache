# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes all platform-specific logic so that
business-logic modules never need to check
``torch.cuda.is_available()`` or ``hasattr(os, "eventfd")``
directly.

Public API::

    from lmcache.v1.platform import (
        HAS_CUDA,
        HAS_EVENTFD,
        EventNotifier,
        MemoryPinner,
        consume_fd,
        create_event_notifier,
        create_memory_pinner,
        cuda_init,
        current_device_id,
        lmc_ops,
        safe_device,
        synchronize,
    )
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.platform.cache_context import (  # noqa: F401
    CacheContextBase,
    CpuCacheContext,
)
from lmcache.v1.platform.capabilities import (  # noqa: F401
    HAS_CUDA,
    HAS_EVENTFD,
)
from lmcache.v1.platform.cuda_utils import (  # noqa: F401
    cuda_init,
    current_device_id,
    safe_device,
    synchronize,
)
from lmcache.v1.platform.event_notifier import (  # noqa: F401
    EventNotifier,
    consume_fd,
    create_event_notifier,
)
from lmcache.v1.platform.gpu_connector import (  # noqa: F401
    MockCudaEvent,
    MockCudaStream,
    create_ipc_event,
    device_guard,
    event_from_ipc_handle,
    mock_memcpy_async_d2h,
    mock_memcpy_async_h2d,
    mock_multi_layer_block_kv_transfer,
    noop_device_guard,
    noop_stream_guard,
    stream_guard,
)
from lmcache.v1.platform.memory_pinner import (  # noqa: F401
    MemoryPinner,
    create_memory_pinner,
)
from lmcache.v1.platform.ops import lmc_ops  # noqa: F401

# ------------------------------------------------------------------
# Unified memcpy & cache-context factory
# ------------------------------------------------------------------
# These resolve the CUDA vs CPU implementation once at import
# time so that callers never need ``if HAS_CUDA`` branches.

if HAS_CUDA:
    # First Party
    from lmcache.v1.gpu_connector.gpu_ops import (  # noqa: F401
        lmcache_memcpy_async_d2h,
        lmcache_memcpy_async_h2d,
    )
    from lmcache.v1.multiprocess.gpu_context import (  # noqa: F401, E501
        GPUCacheContext as _GpuCtxImpl,
    )
else:
    lmcache_memcpy_async_d2h = mock_memcpy_async_d2h  # noqa: F811
    lmcache_memcpy_async_h2d = mock_memcpy_async_h2d  # noqa: F811
    _GpuCtxImpl = None  # type: ignore[assignment,misc]


def create_cache_context(
    chunk_size: int,
    kv_caches: Any = None,
    layout_hints: Any = None,
) -> CacheContextBase:
    """Create the appropriate cache context for the platform.

    On CUDA platforms, *kv_caches* and *layout_hints* are
    forwarded to ``GPUCacheContext``.  On CPU-only platforms a
    ``CpuCacheContext`` is returned; extra keys in
    *layout_hints* (``num_layers``, ``num_heads``,
    ``head_size``, ``num_blocks``, ``block_size``, ``dtype``)
    are forwarded to :class:`CpuCacheContext`.
    """
    if HAS_CUDA and _GpuCtxImpl is not None:
        return _GpuCtxImpl(  # type: ignore[arg-type]
            kv_caches,
            chunk_size,
            layout_hints=layout_hints or None,
        )

    # CPU mode — extract optional config from layout_hints
    cpu_kwargs: dict[str, Any] = {}
    _CPU_HINT_KEYS = (
        "num_layers",
        "num_heads",
        "head_size",
        "num_blocks",
        "block_size",
        "dtype",
    )
    if isinstance(layout_hints, dict):
        for k in _CPU_HINT_KEYS:
            if k in layout_hints:
                cpu_kwargs[k] = layout_hints[k]

    # dtype may arrive as a string ("float16") over the
    # wire because torch.dtype is not serializable.
    if "dtype" in cpu_kwargs and isinstance(cpu_kwargs["dtype"], str):
        # Third Party
        import torch  # noqa: F811

        _DTYPE_MAP: dict[str, Any] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        cpu_kwargs["dtype"] = _DTYPE_MAP.get(cpu_kwargs["dtype"], torch.float16)

    return CpuCacheContext(chunk_size=chunk_size, **cpu_kwargs)
