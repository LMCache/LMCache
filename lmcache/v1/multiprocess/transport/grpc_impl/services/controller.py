# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``ControllerService`` surface."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementService
from lmcache.v1.multiprocess.protocol import HandlerType, grpc_method


class ControllerServiceImpl:
    """Implementation of the generated ``ControllerService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

    @grpc_method(HandlerType.BLOCKING)
    def Clear(self) -> None:
        """Clear all stored KV cache data."""
        return self._management.clear()

    def GetChunkSize(self) -> int:
        """Return the configured chunk size."""
        return self._management.get_chunk_size()

    def GetExperimental(self) -> list[str]:
        """Return enabled experimental server features."""
        return self._management.get_experimental()

    @grpc_method(HandlerType.BLOCKING)
    def Ping(self, instance_id: int | None) -> bool:
        """Refresh worker liveness and report server reachability."""
        return self._management.ping(instance_id)
