# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``ControllerService`` surface."""

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
)


class ControllerServiceImpl:
    """Implement controller RPCs with the management module."""

    def __init__(self, management: ManagementModule) -> None:
        self._management = management

    @grpc_method(GrpcHandlerType.BLOCKING)
    def Clear(self) -> None:
        """Clear all stored KV cache data."""
        return self._management.clear()

    def GetChunkSize(self) -> int:
        """Return the configured chunk size."""
        return self._management.get_chunk_size()

    def GetExperimental(self) -> list[str]:
        """Return enabled experimental server features."""
        return self._management.get_experimental()

    @grpc_method(GrpcHandlerType.BLOCKING)
    def Ping(self, instance_id: int | None) -> bool:
        """Refresh worker liveness and report server reachability."""
        return self._management.ping(instance_id)
