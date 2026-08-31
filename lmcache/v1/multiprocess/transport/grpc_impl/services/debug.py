# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``DebugService`` surface."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementService


class DebugServiceImpl:
    """Implementation of the generated ``DebugService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

    def Noop(self) -> str:
        """Return a simple health-check string."""
        return self._management.debug()
