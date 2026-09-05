# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``LookupService`` surface."""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    grpc_method,
)


class LookupServiceImpl:
    """Implement lookup RPCs with the lookup module."""

    def __init__(self, lookup: LookupModule) -> None:
        self._lookup = lookup

    @grpc_method(GrpcHandlerType.BLOCKING)
    def Lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Start a prefix lookup and prefetch."""
        return self._lookup.lookup(key, tp_size)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def QueryPrefetchStatus(self, request_id: str) -> int | None:
        """Poll prefix prefetch completion."""
        return self._lookup.query_prefetch_status(request_id)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def WaitPrefetchStatus(self, request_id: str, timeout: float) -> int | None:
        """Wait for prefix prefetch completion."""
        return self._lookup.wait_prefetch_status(request_id, timeout)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def QueryPrefetchLookupHits(self, request_id: str) -> int | None:
        """Return a prefetch lookup hit count."""
        return self._lookup.query_prefetch_lookup_hits(request_id)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def FreeLookupLocks(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Release lookup read locks."""
        return self._lookup.free_lookup_locks(key, tp_size)

    @grpc_method(GrpcHandlerType.BLOCKING)
    def EndSession(self, request_id: str) -> None:
        """End a cache session."""
        return self._lookup.end_session(request_id)
