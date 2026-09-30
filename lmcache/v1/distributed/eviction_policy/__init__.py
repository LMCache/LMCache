# SPDX-License-Identifier: Apache-2.0
"""
Eviction policies for L1 cache management
"""

# First Party
from lmcache.v1.distributed.eviction_policy.arc import (
    ARCEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.factory import (
    CreateEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.lru import (
    LRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.noop import (
    NoOpEvictionPolicy,
)

__all__ = [
    "ARCEvictionPolicy",
    "LRUEvictionPolicy",
    "NoOpEvictionPolicy",
    "IsolatedLRUEvictionPolicy",
    "CreateEvictionPolicy",
]
