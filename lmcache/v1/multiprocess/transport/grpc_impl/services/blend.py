# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``BlendService`` surface."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
    require_service,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.modules.blend import BlendModule


class BlendServiceImpl:
    """Implement CacheBlend RPCs when the blend module is enabled."""

    def __init__(self, blend: BlendModule | None) -> None:
        self._blend = blend

    def CbRegisterRope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> None:
        """Register rope metadata for CacheBlend re-RoPE."""
        return require_service(self._blend, "CacheBlend").cb_register_rope(
            instance_id,
            cos_sin_caches_ipc,
            head_size,
            is_neox_style,
            group_to_cache,
            group_rot,
        )

    def CbUnregisterRope(self, instance_id: int) -> None:
        """Unregister CacheBlend rope metadata."""
        return require_service(self._blend, "CacheBlend").cb_unregister_rope(
            instance_id
        )

    @grpc_method(GrpcHandlerType.BLOCKING, requires_client_affinity=True)
    def CbRetrievePreComputed(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Retrieve precomputed CacheBlend chunks."""
        return require_service(self._blend, "CacheBlend").cb_retrieve_pre_computed(
            key, cb_match_result, gpu_block_ids, instance_id, event_ipc_handle
        )

    @grpc_method(GrpcHandlerType.BLOCKING)
    def CbUnifiedLookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> CBUnifiedLookupResult | None:
        """Run the CacheBlend unified lookup."""
        return require_service(self._blend, "CacheBlend").cb_unified_lookup(
            key, tp_size
        )
