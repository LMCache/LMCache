# SPDX-License-Identifier: Apache-2.0
"""Numba JIT helpers shared by LMCache kernels."""

# Standard
from typing import Any, Callable

# Third Party
from numba import njit

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_cache_fallback_warned = False


def njit_cached(func: Callable[..., Any]) -> Callable[..., Any]:
    """Compile ``func`` with ``numba.njit``, using the on-disk cache if possible.

    Numba selects the cache directory while the decorator runs. When no
    writable directory exists it raises ``RuntimeError`` and the import
    fails. This decorator compiles without the disk cache in that case, so
    the compiled code stays in the current process and is rebuilt on first
    use in every new process. A warning is logged once per process. Set
    ``NUMBA_CACHE_DIR`` to a writable directory to keep the disk cache.

    Args:
        func: Function to compile in nopython mode.

    Returns:
        The Numba dispatcher wrapping ``func``.
    """
    global _cache_fallback_warned
    try:
        return njit(cache=True)(func)
    except RuntimeError as exc:
        if not _cache_fallback_warned:
            _cache_fallback_warned = True
            logger.warning(
                "Numba disk caching is unavailable (%s). Falling back to "
                "cache=False; JIT-compiled functions will be recompiled in "
                "each new process. Set NUMBA_CACHE_DIR to a writable "
                "directory to restore disk caching.",
                exc,
            )
        return njit(cache=False)(func)
