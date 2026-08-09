# SPDX-License-Identifier: Apache-2.0
"""Top-level re-export of the compiled ``lmcache.lmcache_native`` module.

The native operations live in the ``lmcache.lmcache_native`` extension (built
from ``csrc/lmcache_native/pybind.cpp``). This shim lets callers use the
shorter, backend-agnostic ``import lmcache_native`` while the canonical module
remains ``lmcache.lmcache_native``.
"""

# Standard
import sys

# The native extension may be absent in CPU-only / test environments. Rather
# than failing the whole import (which would break ``install_lmcache_native_
# fallback`` in test conftest), fall back to a thin module so callers and the
# test fallback hook (which installs ``sys.modules["lmcache.lmcache_native"]``)
# can recover.
try:
    # First Party
    import lmcache.lmcache_native as _native

    sys.modules[__name__] = _native
except Exception:
    pass
