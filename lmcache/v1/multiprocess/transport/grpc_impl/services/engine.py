# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``EngineService`` surface."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferService,
)
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreService
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferService,
)
from lmcache.v1.multiprocess.modules.lookup import EngineLookupService
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mp_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl.protocol import (
    HandlerType,
    grpc_method,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.modules.blend import BlendService


T = TypeVar("T")

# Generated protobuf classes are dynamic and opaque to static analysis.
lmcache_mp_pb2: Any = _pb2_typed


class _StoreService(Protocol):
    """Store-capable implementation backing ``EngineService.Store``."""

    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store KV chunks for an engine worker."""
        ...


def _require(service: T | None, feature: str) -> T:
    """Return ``service`` or raise gRPC UNIMPLEMENTED through the transport."""
    if service is None:
        raise NotImplementedError(f"{feature} is not enabled on this server")
    return service


class EngineServiceImpl:
    """Implementation of the generated ``EngineService`` RPC surface."""

    def __init__(
        self,
        lookup: EngineLookupService,
        *,
        lmcache_driven_transfer: LMCacheDrivenTransferService | None = None,
        engine_driven_transfer: EngineDrivenTransferService | None = None,
        qstore: QStoreService | None = None,
        blend: BlendService | None = None,
    ) -> None:
        self._lookup = lookup
        self._lmcache_driven_transfer = lmcache_driven_transfer
        self._engine_driven_transfer = engine_driven_transfer
        self._qstore = qstore
        self._store_service: _StoreService | None = (
            blend if blend is not None else lmcache_driven_transfer
        )

    def RegisterKvCache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register an LMCache-driven KV cache context."""
        _require(
            self._lmcache_driven_transfer, "LMCache-driven KV transfer"
        ).register_kv_cache(
            instance_id,
            kv_caches,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def RegisterQCache(
        self,
        instance_id: int,
        q_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register an experimental paged-Q cache context."""
        _require(self._qstore, "QStore transfer").register_q_cache(
            instance_id,
            q_caches,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    def UnregisterKvCache(self, instance_id: int) -> None:
        """Unregister an LMCache-driven KV cache context."""
        _require(
            self._lmcache_driven_transfer, "LMCache-driven KV transfer"
        ).unregister_kv_cache(instance_id)

    def UnregisterQCache(self, instance_id: int) -> None:
        """Unregister an experimental paged-Q cache context."""
        _require(self._qstore, "QStore transfer").unregister_q_cache(instance_id)

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def StoreQ(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store experimental paged-Q blocks."""
        return _require(self._qstore, "QStore transfer").store_q(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def Store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store KV blocks, using the blend store path when enabled."""
        return _require(self._store_service, "LMCache-driven KV transfer").store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def Retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> tuple[bytes, bool]:
        """Retrieve KV blocks into an LMCache-driven cache context."""
        return _require(
            self._lmcache_driven_transfer, "LMCache-driven KV transfer"
        ).retrieve(
            key, instance_id, gpu_block_ids, event_ipc_handle, skip_first_n_tokens
        )

    @grpc_method(HandlerType.BLOCKING)
    def Lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Start a prefix lookup/prefetch for ``key``."""
        return self._lookup.lookup(key, tp_size)

    @grpc_method(HandlerType.BLOCKING)
    def QueryPrefetchStatus(self, request_id: str) -> int | None:
        """Poll prefix prefetch completion for ``request_id``."""
        return self._lookup.query_prefetch_status(request_id)

    @grpc_method(HandlerType.BLOCKING)
    def WaitPrefetchStatus(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Wait for prefix prefetch completion for ``request_id``."""
        return self._lookup.wait_prefetch_status(request_id, timeout)

    @grpc_method(HandlerType.BLOCKING)
    def QueryPrefetchLookupHits(self, request_id: str) -> int | None:
        """Return the lookup hit count for a completed prefetch."""
        return self._lookup.query_prefetch_lookup_hits(request_id)

    @grpc_method(HandlerType.BLOCKING)
    def FreeLookupLocks(self, key: IPCCacheServerKey, n: int) -> None:
        """Release lookup locks held for ``key``."""
        return self._lookup.free_lookup_locks(key, n)

    @grpc_method(HandlerType.BLOCKING)
    def EndSession(self, request_id: str) -> None:
        """End an active request session."""
        return self._lookup.end_session(request_id)

    def RegisterKvCacheEngineDrivenContext(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> lmcache_mp_pb2.RegisterKvCacheEngineDrivenContextResponse:
        """Register an engine-driven KV context."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).register_kv_cache_engine_driven_context(payload)

    def UnregisterKvCacheEngineDrivenContext(self, instance_id: int) -> None:
        """Unregister an engine-driven KV context."""
        _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).unregister_kv_cache(instance_id)

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def PrepareStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> lmcache_mp_pb2.PrepareStoreResponse:
        """Prepare an engine-driven store."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).prepare_store(key, instance_id)

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def CommitStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        data: bytes,
    ) -> bool:
        """Commit an engine-driven store."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).commit_store(key, instance_id, data)

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def PrepareRetrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> lmcache_mp_pb2.PrepareRetrieveResponse:
        """Prepare an engine-driven retrieve."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).prepare_retrieve(key, instance_id)

    @grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
    def CommitRetrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit an engine-driven retrieve."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).commit_retrieve(key, instance_id)
