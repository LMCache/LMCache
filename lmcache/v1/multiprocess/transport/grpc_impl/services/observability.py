# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``ObservabilityService`` surface."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.modules.management import ManagementService
from lmcache.v1.multiprocess.protocol import HandlerType, grpc_method


class ObservabilityServiceImpl:
    """Implementation of the generated ``ObservabilityService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

    @grpc_method(HandlerType.BLOCKING)
    def ReportBlockAllocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the event bus."""
        return self._management.report_block_allocations(
            instance_id, model_name, records
        )
