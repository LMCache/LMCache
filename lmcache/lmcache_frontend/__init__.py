# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the moved fleet frontend.

The fleet frontend now lives at :mod:`lmcache.v1.server.frontend`. This
package re-exports its submodules so existing ``lmcache.lmcache_frontend``
imports keep working until the old path is separately deprecated.
"""

# Standard
import sys

# First Party
from lmcache.v1.server.frontend import app, heartbeat, lmcache_mp_plugin
from lmcache.v1.server.frontend.lmcache_mp_plugin import (
    lmcache_mp_frontend_plugin,
)

__all__ = [
    "app",
    "heartbeat",
    "lmcache_mp_plugin",
]

# Point the legacy submodule paths at the canonical modules so
# ``from lmcache.lmcache_frontend.<name> import ...`` keeps working.
sys.modules[__name__ + ".app"] = app
sys.modules[__name__ + ".heartbeat"] = heartbeat
sys.modules[__name__ + ".lmcache_mp_plugin"] = lmcache_mp_plugin
sys.modules[__name__ + ".lmcache_mp_plugin.lmcache_mp_frontend_plugin"] = (
    lmcache_mp_frontend_plugin
)
