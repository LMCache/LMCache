# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``DebugService`` surface."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.modules.management import ManagementModule


class DebugServiceImpl:
    """Implement debug RPCs with the management module."""

    def __init__(self, management: ManagementModule) -> None:
        self._management = management

    def Noop(self) -> str:
        """Return a simple health-check string."""
        return self._management.debug()
