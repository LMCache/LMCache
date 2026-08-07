# SPDX-License-Identifier: Apache-2.0
"""Top-level re-export of the compiled ``lmcache.lmache_native`` module.

The native operations live in the ``lmcache.lmache_native`` extension (built
from ``csrc/storage_manager/pybind.cpp``). This shim lets callers use the
shorter, backend-agnostic ``import lmache_native`` while the canonical module
remains ``lmcache.lmache_native``.
"""

# Standard
import sys

import lmcache.lmache_native as _native

sys.modules[__name__] = _native
