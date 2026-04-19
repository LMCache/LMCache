# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes platform-specific logic.
"""

# First Party
from lmcache.v1.platform.eventfd_compat import install_eventfd_compat

# Safety net: patch os.eventfd on non-Linux platforms so that
# call-sites can keep using ``os.eventfd`` transparently.
install_eventfd_compat()
