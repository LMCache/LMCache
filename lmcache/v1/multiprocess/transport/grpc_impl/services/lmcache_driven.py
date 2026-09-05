# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``LMCacheDrivenService`` surface."""

# Standard
from typing import Protocol

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
    require_service,
)


class _StoreModule(Protocol):
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store KV chunks for an engine worker."""
        ...


class LMCacheDrivenServiceImpl:
    """Implement LMCache-driven transfer RPCs."""

    def __init__(
        self,
        transfer: LMCacheDrivenTransferModule | None,
        store_module: _StoreModule | None,
    ) -> None:
        self._transfer = transfer
        self._store_module = store_module

    def RegisterKvCache(
        self,
        instance_id: int,
        kv_cache: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register an LMCache-driven KV cache context."""
        return require_service(
            self._transfer, "LMCache-driven transfer"
        ).register_kv_cache(
            instance_id,
            kv_cache,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def UnregisterKvCache(self, instance_id: int) -> None:
        """Unregister an LMCache-driven KV cache context."""
        return require_service(
            self._transfer, "LMCache-driven transfer"
        ).unregister_kv_cache(instance_id)

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def Store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store KV blocks, using the blend path when enabled."""
        return require_service(self._store_module, "LMCache-driven transfer").store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def Retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> tuple[bytes, bool]:
        """Retrieve KV blocks into an LMCache-driven cache context."""
        return require_service(self._transfer, "LMCache-driven transfer").retrieve(
            key,
            instance_id,
            gpu_block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        )
