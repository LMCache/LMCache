# SPDX-License-Identifier: Apache-2.0
"""L1 managers: one object-lifecycle state machine, one subclass per medium.

- :class:`DramL1Manager`, :class:`DevDaxL1Manager` and :class:`GdsL1Manager`
  implement :class:`~lmcache.v1.distributed.internal_api.L1ManagerInterface`
  for one memory medium each, sharing the state machine in ``base.py``
  (an implementation detail of this package).
- :func:`create_l1_manager` / :func:`create_l1_managers` pick the subclass
  from an :class:`~lmcache.v1.distributed.config.L1ManagerConfig`.
"""

# First Party
from lmcache.v1.distributed.l1_manager.base import (
    MAX_READ_LOCK_COUNT,
    L1ObjectState,
    l1_mgr_synchronized,
    next_l1_manager_id,
    validate_read_locks,
)
from lmcache.v1.distributed.l1_manager.devdax import DevDaxL1Manager
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.l1_manager.factory import (
    create_l1_manager,
    create_l1_managers,
)
from lmcache.v1.distributed.l1_manager.gds import GdsL1Manager

__all__ = [
    "MAX_READ_LOCK_COUNT",
    "DevDaxL1Manager",
    "DramL1Manager",
    "GdsL1Manager",
    "L1ObjectState",
    "create_l1_manager",
    "create_l1_managers",
    "l1_mgr_synchronized",
    "next_l1_manager_id",
    "validate_read_locks",
]
