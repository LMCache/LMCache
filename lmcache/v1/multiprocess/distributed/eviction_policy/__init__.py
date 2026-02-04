# SPDX-License-Identifier: Apache-2.0
"""
Eviction policies for L1 cache management
"""

# First Party
from lmcache.v1.multiprocess.distributed.eviction_policy.factory import (
    CreateEvictionPolicy,
)
from lmcache.v1.multiprocess.distributed.eviction_policy.lru import (
    LRUEvictionPolicy,
)

__all__ = [
    "LRUEvictionPolicy",
    "CreateEvictionPolicy",
]
