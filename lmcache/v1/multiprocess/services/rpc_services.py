# SPDX-License-Identifier: Apache-2.0
"""Concrete Python implementations for the generated LMCache gRPC services."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Protocol, TypeVar

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    KVCache,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.services.engine_driven_transfer import (
    EngineDrivenTransferService,
)
from lmcache.v1.multiprocess.services.experimental.qstore import QStoreService
from lmcache.v1.multiprocess.services.lmcache_driven_transfer import (
    LMCacheDrivenTransferService,
)
from lmcache.v1.multiprocess.services.lookup import EngineLookupService
from lmcache.v1.multiprocess.services.management import ManagementService
from lmcache.v1.multiprocess.services.p2p_controller import P2PController

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.services.blend import LegacyBlendService
    from lmcache.v1.multiprocess.services.blend_v3 import BlendV3Service


T = TypeVar("T")


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
        blend_v3: BlendV3Service | None = None,
    ) -> None:
        self._lookup = lookup
        self._lmcache_driven_transfer = lmcache_driven_transfer
        self._engine_driven_transfer = engine_driven_transfer
        self._qstore = qstore
        self._store_service: _StoreService | None = (
            blend_v3 if blend_v3 is not None else lmcache_driven_transfer
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

    def Store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store KV blocks, using the blend V3 store path when enabled."""
        return _require(self._store_service, "LMCache-driven KV transfer").store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

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

    def Lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Start a prefix lookup/prefetch for ``key``."""
        return self._lookup.lookup(key, tp_size)

    def QueryPrefetchStatus(self, request_id: str) -> int | None:
        """Poll prefix prefetch completion for ``request_id``."""
        return self._lookup.query_prefetch_status(request_id)

    def WaitPrefetchStatus(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Wait for prefix prefetch completion for ``request_id``."""
        return self._lookup.wait_prefetch_status(request_id, timeout)

    def QueryPrefetchLookupHits(self, request_id: str) -> int | None:
        """Return the lookup hit count for a completed prefetch."""
        return self._lookup.query_prefetch_lookup_hits(request_id)

    def FreeLookupLocks(self, key: IPCCacheServerKey, n: int) -> None:
        """Release lookup locks held for ``key``."""
        return self._lookup.free_lookup_locks(key, n)

    def EndSession(self, request_id: str) -> None:
        """End an active request session."""
        return self._lookup.end_session(request_id)

    def RegisterKvCacheEngineDrivenContext(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register an engine-driven KV context."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).register_kv_cache_engine_driven_context(payload)

    def UnregisterKvCacheEngineDrivenContext(self, instance_id: int) -> None:
        """Unregister an engine-driven KV context."""
        _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).unregister_kv_cache(instance_id)

    def PrepareStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare an engine-driven store."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).prepare_store(key, instance_id)

    def CommitStore(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit an engine-driven store."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).commit_store(key, instance_id, cpu_data)

    def PrepareRetrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Prepare an engine-driven retrieve."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).prepare_retrieve(key, instance_id)

    def CommitRetrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit an engine-driven retrieve."""
        return _require(
            self._engine_driven_transfer, "engine-driven KV transfer"
        ).commit_retrieve(key, instance_id)


class ControllerServiceImpl:
    """Implementation of the generated ``ControllerService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

    def Clear(self) -> None:
        """Clear all stored KV cache data."""
        return self._management.clear()

    def GetChunkSize(self) -> int:
        """Return the configured chunk size."""
        return self._management.get_chunk_size()

    def GetExperimental(self) -> list[str]:
        """Return enabled experimental server features."""
        return self._management.get_experimental()

    def Ping(self, instance_id: int | None) -> bool:
        """Refresh worker liveness and report server reachability."""
        return self._management.ping(instance_id)


class DebugServiceImpl:
    """Implementation of the generated ``DebugService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

    def Noop(self) -> str:
        """Return a simple health-check string."""
        return self._management.debug()


class ObservabilityServiceImpl:
    """Implementation of the generated ``ObservabilityService`` RPC surface."""

    def __init__(self, management: ManagementService) -> None:
        self._management = management

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


class P2PServiceImpl:
    """Implementation of the generated ``P2PService`` RPC surface."""

    def __init__(self, controller: P2PController) -> None:
        self._controller = controller

    def P2PLookupAndLock(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        """Start a peer-to-peer lookup and lock matching L1 objects."""
        return self._controller.p2p_lookup_and_lock(keys, group_layout_descs)

    def P2PQueryLookupResults(
        self,
        task_id: int,
    ) -> list[TransferChannelAddress] | None:
        """Poll the result of a peer-to-peer lookup."""
        return self._controller.p2p_query_lookup_results(task_id)

    def P2PUnlockObjects(self, keys: list[ObjectKey]) -> None:
        """Release peer-to-peer read locks for object keys."""
        return self._controller.p2p_unlock_objects(keys)


class BlendServiceImpl:
    """Implementation of the generated legacy ``BlendService`` RPC surface."""

    def __init__(self, blend: LegacyBlendService) -> None:
        self._blend = blend

    def CbRegisterKvCache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
    ) -> None:
        """Register a legacy CacheBlend KV cache context."""
        return self._blend.cb_register_kv_cache(
            instance_id, kv_caches, model_name, world_size
        )

    def CbUnregisterKvCache(self, instance_id: int) -> None:
        """Unregister a legacy CacheBlend KV cache context."""
        return self._blend.cb_unregister_kv_cache(instance_id)

    def CbStorePreComputed(
        self,
        key: IPCCacheServerKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store pre-computed legacy CacheBlend chunks."""
        return self._blend.cb_store_pre_computed(
            key, offset, instance_id, event_ipc_handle
        )

    def CbLookupPreComputed(
        self,
        key: IPCCacheServerKey,
    ) -> list[tuple[int, int]]:
        """Reject the superseded legacy lookup RPC."""
        del key
        raise NotImplementedError(
            "BlendService.CbLookupPreComputed is superseded by "
            "BlendV2Service.CbLookupPreComputedV2"
        )

    def CbRetrievePreComputed(
        self,
        key: IPCCacheServerKey,
        token_ranges: list[tuple[int, int]],
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Reject the superseded legacy retrieve RPC."""
        del key, token_ranges, offset, instance_id, event_ipc_handle
        raise NotImplementedError(
            "BlendService.CbRetrievePreComputed is superseded by "
            "BlendV2Service.CbRetrievePreComputedV2"
        )

    def CbStoreFinal(
        self,
        key: IPCCacheServerKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store final legacy CacheBlend chunks."""
        return self._blend.cb_store_final(key, offset, instance_id, event_ipc_handle)


class BlendV2ServiceImpl:
    """Implementation of the generated ``BlendV2Service`` RPC surface."""

    def __init__(self, blend: LegacyBlendService) -> None:
        self._blend = blend

    def CbLookupPreComputedV2(self, key: IPCCacheServerKey) -> list[CBMatchResult]:
        """Lookup pre-computed CacheBlend chunks."""
        return self._blend.cb_lookup_pre_computed(key)

    def CbRetrievePreComputedV2(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Retrieve pre-computed CacheBlend chunks."""
        return self._blend.cb_retrieve_pre_computed(
            key, cb_match_result, offset, instance_id, event_ipc_handle
        )


class BlendV3ServiceImpl:
    """Implementation of the generated ``BlendV3Service`` RPC surface."""

    def __init__(self, blend: BlendV3Service) -> None:
        self._blend = blend

    def CbRegisterRopeV3(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> None:
        """Register rope metadata for CacheBlend V3 re-RoPE."""
        return self._blend.cb_register_rope(
            instance_id,
            cos_sin_caches_ipc,
            head_size,
            is_neox_style,
            group_to_cache,
            group_rot,
        )

    def CbUnregisterRopeV3(self, instance_id: int) -> None:
        """Unregister CacheBlend V3 rope metadata."""
        return self._blend.cb_unregister_rope(instance_id)

    def CbRetrievePreComputedV3(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Retrieve pre-computed CacheBlend V3 chunks."""
        return self._blend.cb_retrieve_pre_computed(
            key, cb_match_result, gpu_block_ids, instance_id, event_ipc_handle
        )

    def CbUnifiedLookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> CBUnifiedLookupResult | None:
        """Run the CacheBlend V3 unified lookup RPC."""
        return self._blend.cb_unified_lookup(key, tp_size)
