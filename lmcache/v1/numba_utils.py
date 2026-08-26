# SPDX-License-Identifier: Apache-2.0
"""Numba JIT helpers with a graceful disk-cache fallback."""

# Standard
from typing import Callable

# Third Party
from numba import njit

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_cache_fallback_warned = False


def njit_cached(func: Callable) -> Callable:
    """Compile ``func`` with ``njit(cache=True)``, falling back to
    ``cache=False`` (with a one-time warning) when no writable cache
    directory is available.

    Args:
        func: The Python function to JIT-compile.

    Returns:
        The numba-compiled dispatcher wrapping ``func``.
    """
    try:
        return njit(cache=True)(func)
    except RuntimeError as exc:
        global _cache_fallback_warned
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
