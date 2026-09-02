# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``QStoreService`` surface."""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
    require_service,
)


class QStoreServiceImpl:
    """Implement experimental QStore RPCs when enabled."""

    def __init__(self, qstore: QStoreModule | None) -> None:
        self._qstore = qstore

    def RegisterQCache(
        self,
        instance_id: int,
        kv_cache: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register an experimental paged-Q cache context."""
        return require_service(self._qstore, "QStore").register_q_cache(
            instance_id,
            kv_cache,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def UnregisterQCache(self, instance_id: int) -> None:
        """Unregister an experimental paged-Q cache context."""
        return require_service(self._qstore, "QStore").unregister_q_cache(instance_id)

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def StoreQ(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store experimental paged-Q blocks."""
        return require_service(self._qstore, "QStore").store_q(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )
