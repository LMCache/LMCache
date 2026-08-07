# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible alias for :mod:`lmcache.lmache_native`.

``native_storage_ops`` was renamed to ``lmache_native`` to better reflect its
scope (device-independent KV-format/transfer types, not just storage ops).
Importing this module now resolves to the renamed extension. New code should
import :mod:`lmcache.lmache_native` (or the top-level :mod:`lmache_native`)
directly.
"""

# Standard
import sys

import lmcache.lmache_native as _native

sys.modules[__name__] = _native
