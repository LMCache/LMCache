# SPDX-License-Identifier: Apache-2.0
"""
LMCache v1 API.

Importing this module ensures the crash handler is installed.
"""

# First Party
# Ensure the crash handler is installed when v1 is imported
# This handles cases where users do: from lmcache.v1 import ...
import lmcache  # noqa: F401
