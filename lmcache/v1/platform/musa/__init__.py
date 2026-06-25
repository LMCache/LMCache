# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform primitives.

The MUSA IPC wrapper is discovered from
:mod:`lmcache.v1.platform.musa.ipc` by the platform registry. Importing this
package only registers the explicit availability predicate so MUSA workers keep
using the Stage3 data path unless the Stage4 handle path is fully available and
explicitly enabled.
"""

# First Party
from lmcache.v1.platform._registry import register_availability


def _musa_handle_transfer_is_available() -> bool:
    """Return whether the optional MUSA handle path can be selected."""
    # First Party
    from lmcache.v1.platform.musa.ipc import is_musa_handle_transfer_available

    return is_musa_handle_transfer_available()


register_availability("musa", _musa_handle_transfer_is_available)
