# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``P2PService`` surface."""

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
)


class P2PServiceImpl:
    """Implement peer transfer RPCs with the P2P controller."""

    def __init__(self, controller: P2PController) -> None:
        self._controller = controller

    @grpc_method(GrpcHandlerType.BLOCKING)
    def P2PLookupAndLock(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        """Start a peer lookup and lock matching objects."""
        return self._controller.p2p_lookup_and_lock(keys, group_layout_descs)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def P2PQueryLookupResults(
        self, task_id: int
    ) -> list[TransferChannelAddress] | None:
        """Poll the result of a peer lookup."""
        return self._controller.p2p_query_lookup_results(task_id)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def P2PUnlockObjects(self, keys: list[ObjectKey]) -> None:
        """Release peer read locks for object keys."""
        return self._controller.p2p_unlock_objects(keys)
