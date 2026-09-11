# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for LMCache-driven transfer RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class EventIpcHandleResult:
    """Result of a device transfer submission."""

    event_ipc_handle: bytes
    success: bool


@dataclass(frozen=True)
class RegisterKvCacheRequest:
    """Register an LMCache-driven worker cache."""

    instance_id: int
    kv_cache: KVCache
    model_name: str
    world_size: int
    engine_type: EngineType
    layout_hints: LayoutHints
    engine_group_infos: list[EngineGroupInfo]


@dataclass(frozen=True)
class RegisterKvCacheResponse:
    """Acknowledge worker-cache registration."""


@dataclass(frozen=True)
class UnregisterKvCacheRequest:
    """Unregister an LMCache-driven worker cache."""

    instance_id: int


@dataclass(frozen=True)
class UnregisterKvCacheResponse:
    """Acknowledge worker-cache removal."""


@dataclass(frozen=True)
class StoreRequest:
    """Store device blocks in LMCache."""

    key: IPCCacheServerKey
    instance_id: int
    gpu_block_ids: list[list[int]]
    event_ipc_handle: bytes


@dataclass(frozen=True)
class StoreResponse:
    """Return the completion event for a store submission."""

    result: EventIpcHandleResult


@dataclass(frozen=True)
class RetrieveRequest:
    """Retrieve cached data into device blocks."""

    key: IPCCacheServerKey
    instance_id: int
    gpu_block_ids: list[list[int]]
    event_ipc_handle: bytes
    skip_first_n_tokens: int


@dataclass(frozen=True)
class RetrieveResponse:
    """Return the completion event for a retrieve submission."""

    result: EventIpcHandleResult


register_rpc_message_types(
    RequestType.REGISTER_KV_CACHE, RegisterKvCacheRequest, RegisterKvCacheResponse
)
register_rpc_message_types(
    RequestType.UNREGISTER_KV_CACHE,
    UnregisterKvCacheRequest,
    UnregisterKvCacheResponse,
)
register_rpc_message_types(RequestType.STORE, StoreRequest, StoreResponse)
register_rpc_message_types(RequestType.RETRIEVE, RetrieveRequest, RetrieveResponse)
# QStore owns distinct operations but deliberately shares the LMCache-driven
# payload shapes.  Keep those aliases beside the payload classes, not in a
# repository-wide protocol table.
register_rpc_message_types(
    RequestType.REGISTER_Q_CACHE, RegisterKvCacheRequest, RegisterKvCacheResponse
)
register_rpc_message_types(
    RequestType.UNREGISTER_Q_CACHE,
    UnregisterKvCacheRequest,
    UnregisterKvCacheResponse,
)
register_rpc_message_types(RequestType.STORE_Q, StoreRequest, StoreResponse)


__all__ = [
    "EventIpcHandleResult",
    "RegisterKvCacheRequest",
    "RegisterKvCacheResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "StoreRequest",
    "StoreResponse",
    "UnregisterKvCacheRequest",
    "UnregisterKvCacheResponse",
]
