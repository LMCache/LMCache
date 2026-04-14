# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes all platform-specific logic.
"""

# First Party
from lmcache.v1.platform.cuda_compat import install_cuda_compat
from lmcache.v1.platform.cupy_compat import install_cupy_compat
from lmcache.v1.platform.eventfd_compat import (
    install_eventfd_compat,
)

# Safety net: patch torch.cuda on CPU-only platforms so that
# any stray ``torch.cuda.xxx()`` call is a harmless no-op.
install_cuda_compat()

# Safety net: inject a fake cupy module on CPU-only platforms
# so that ``import cupy`` succeeds with harmless no-ops.
install_cupy_compat()

# Safety net: patch os.eventfd on non-Linux platforms so that
# call-sites can keep using ``os.eventfd`` transparently.
install_eventfd_compat()


def __getattr__(name: str) -> object:
    """Lazy re-export of platform cache utilities.

    Deferred so that ``lmcache.c_ops`` (a compiled extension)
    is fully available by the time the class is first used.
    """
    if name == "CpuCacheContext":
        # First Party
        from lmcache.v1.platform.cache_context import (
            CpuCacheContext,
        )

        return CpuCacheContext

    if name == "create_cache_context":
        # First Party
        from lmcache.v1.platform.cache_context import (
            create_cache_context,
        )

        return create_cache_context

    raise AttributeError(name)
