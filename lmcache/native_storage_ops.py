# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible alias for :mod:`lmcache.lmcache_native`.

``native_storage_ops`` was renamed to ``lmcache_native`` to better reflect its
scope (device-independent KV-format/transfer types, not just storage ops).
Importing this module now resolves to the renamed extension. New code should
import :mod:`lmcache.lmcache_native` (or the top-level :mod:`lmcache_native`)
directly.
"""

# Standard
import sys

# First Party
import lmcache.lmcache_native as _native

sys.modules[__name__] = _native
