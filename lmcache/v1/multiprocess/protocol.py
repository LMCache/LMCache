# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral type aliases for LMCache multiprocess mode."""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

# Type aliases kept for callers that share multiprocess payload types without
# depending on a concrete transport implementation.
InstanceID = int
KeyType = IPCCacheServerKey

__all__ = [
    "InstanceID",
    "KeyType",
]
