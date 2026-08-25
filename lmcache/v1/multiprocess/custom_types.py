# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports used by legacy MP integrations."""

# First Party
from lmcache.multiprocess.custom_types import BlockAllocationRecord, CBMatchResult

__all__ = ["BlockAllocationRecord", "CBMatchResult"]
