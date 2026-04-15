# SPDX-License-Identifier: Apache-2.0
"""
Eviction policies for L1 cache management
"""

# First Party
from lmcache.distributed.eviction_policy.factory import (
    CreateEvictionPolicy,
)
from lmcache.distributed.eviction_policy.lru import (
    LRUEvictionPolicy,
)
from lmcache.distributed.eviction_policy.noop import (
    NoOpEvictionPolicy,
)

__all__ = [
    "LRUEvictionPolicy",
    "NoOpEvictionPolicy",
    "CreateEvictionPolicy",
]
