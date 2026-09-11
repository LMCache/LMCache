# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for lookup and prefetch RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class LookupRequest:
    """Start a prefix lookup."""

    key: IPCCacheServerKey
    tp_size: int


@dataclass(frozen=True)
class LookupResponse:
    """Acknowledge lookup submission."""


@dataclass(frozen=True)
class QueryPrefetchStatusRequest:
    """Query completion of a prefetch task."""

    request_id: str


@dataclass(frozen=True)
class QueryPrefetchStatusResponse:
    """Return the completed chunk count, if available."""

    chunk_count: int | None


@dataclass(frozen=True)
class WaitPrefetchStatusRequest:
    """Wait for a prefetch task to complete."""

    request_id: str
    timeout: float


@dataclass(frozen=True)
class WaitPrefetchStatusResponse:
    """Return the completed chunk count, or ``None`` on timeout."""

    chunk_count: int | None


@dataclass(frozen=True)
class QueryPrefetchLookupHitsRequest:
    """Query lookup hits before prefetch completion."""

    request_id: str


@dataclass(frozen=True)
class QueryPrefetchLookupHitsResponse:
    """Return the lookup hit chunk count, if available."""

    chunk_count: int | None


@dataclass(frozen=True)
class FreeLookupLocksRequest:
    """Release lookup read locks."""

    key: IPCCacheServerKey
    tp_size: int


@dataclass(frozen=True)
class FreeLookupLocksResponse:
    """Acknowledge lookup-lock release."""


@dataclass(frozen=True)
class EndSessionRequest:
    """End a lookup session."""

    request_id: str


@dataclass(frozen=True)
class EndSessionResponse:
    """Acknowledge session cleanup."""


register_rpc_message_types("lookup", LookupRequest, LookupResponse)
register_rpc_message_types(
    "query_prefetch_status",
    QueryPrefetchStatusRequest,
    QueryPrefetchStatusResponse,
)
register_rpc_message_types(
    "wait_prefetch_status",
    WaitPrefetchStatusRequest,
    WaitPrefetchStatusResponse,
)
register_rpc_message_types(
    "query_prefetch_lookup_hits",
    QueryPrefetchLookupHitsRequest,
    QueryPrefetchLookupHitsResponse,
)
register_rpc_message_types(
    "free_lookup_locks",
    FreeLookupLocksRequest,
    FreeLookupLocksResponse,
)
register_rpc_message_types("end_session", EndSessionRequest, EndSessionResponse)


__all__ = [
    "EndSessionRequest",
    "EndSessionResponse",
    "FreeLookupLocksRequest",
    "FreeLookupLocksResponse",
    "LookupRequest",
    "LookupResponse",
    "QueryPrefetchLookupHitsRequest",
    "QueryPrefetchLookupHitsResponse",
    "QueryPrefetchStatusRequest",
    "QueryPrefetchStatusResponse",
    "WaitPrefetchStatusRequest",
    "WaitPrefetchStatusResponse",
]
