# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``ObservabilityService`` surface."""

# First Party
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
)


class ObservabilityServiceImpl:
    """Implement observability RPCs with the management module."""

    def __init__(self, management: ManagementModule) -> None:
        self._management = management

    @grpc_method(GrpcHandlerType.BLOCKING)
    def ReportBlockAllocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish block allocation records to the event bus."""
        return self._management.report_block_allocations(
            instance_id, model_name, records
        )
