# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``EngineDrivenService`` surface."""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
    require_service,
)


class EngineDrivenServiceImpl:
    """Implement engine-driven transfer RPCs."""

    def __init__(self, transfer: EngineDrivenTransferModule | None) -> None:
        self._transfer = transfer

    def RegisterKvCacheEngineDrivenContext(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register an engine-driven KV context."""
        return require_service(
            self._transfer, "engine-driven transfer"
        ).register_kv_cache_engine_driven_context(payload)

    def UnregisterKvCacheEngineDrivenContext(self, instance_id: int) -> None:
        """Unregister an engine-driven KV context."""
        return require_service(
            self._transfer, "engine-driven transfer"
        ).unregister_kv_cache(instance_id)

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def PrepareStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare an engine-driven store."""
        return require_service(self._transfer, "engine-driven transfer").prepare_store(
            key, instance_id
        )

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def CommitStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        data: bytes,
    ) -> bool:
        """Commit an engine-driven store."""
        return require_service(self._transfer, "engine-driven transfer").commit_store(
            key, instance_id, data
        )

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def PrepareRetrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Prepare an engine-driven retrieve."""
        return require_service(
            self._transfer, "engine-driven transfer"
        ).prepare_retrieve(key, instance_id)

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def CommitRetrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit an engine-driven retrieve."""
        return require_service(
            self._transfer, "engine-driven transfer"
        ).commit_retrieve(key, instance_id)
