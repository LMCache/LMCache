# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``P2PService`` surface."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.protocol import HandlerType, grpc_method


class P2PServiceImpl:
    """Implementation of the generated ``P2PService`` RPC surface."""

    def __init__(self, controller: P2PController) -> None:
        self._controller = controller

    @grpc_method(HandlerType.BLOCKING)
    def P2PLookupAndLock(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        """Start a peer-to-peer lookup and lock matching L1 objects."""
        return self._controller.p2p_lookup_and_lock(keys, group_layout_descs)

    @grpc_method(HandlerType.BLOCKING)
    def P2PQueryLookupResults(
        self,
        task_id: int,
    ) -> list[TransferChannelAddress] | None:
        """Poll the result of a peer-to-peer lookup."""
        return self._controller.p2p_query_lookup_results(task_id)

    @grpc_method(HandlerType.BLOCKING)
    def P2PUnlockObjects(self, keys: list[ObjectKey]) -> None:
        """Release peer-to-peer read locks for object keys."""
        return self._controller.p2p_unlock_objects(keys)
