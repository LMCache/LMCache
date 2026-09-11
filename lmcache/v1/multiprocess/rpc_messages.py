# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request and response messages for multiprocess RPCs.

Each generated protobuf RPC has a Python message pair in this module.  The
business layer consumes these messages directly; transports are responsible
only for adapting them to their wire representation.
"""

# Standard
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.custom_types import (
    PrepareRetrieveResponse as LegacyPrepareRetrieveResponse,
)
from lmcache.v1.multiprocess.custom_types import (
    PrepareStoreResponse as LegacyPrepareStoreResponse,
)
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextResponse as LegacyRegisterContextResponse,
)
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

MessageT = TypeVar("MessageT")


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


@dataclass(frozen=True)
class RegisterKvCacheEngineDrivenContextRequest:
    """Register an engine-driven transfer context."""

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    num_physical_slots: int | None = None


@dataclass(frozen=True)
class RegisterKvCacheEngineDrivenContextResponse:
    """Return the shared-memory allocation for an engine-driven context."""

    shm_name: str = ""
    pool_size: int = 0


@dataclass(frozen=True)
class UnregisterKvCacheEngineDrivenContextRequest:
    """Unregister an engine-driven transfer context."""

    instance_id: int


@dataclass(frozen=True)
class UnregisterKvCacheEngineDrivenContextResponse:
    """Acknowledge engine-driven context removal."""


@dataclass(frozen=True)
class PrepareStoreRequest:
    """Prepare an engine-driven store operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class PrepareStoreResponse:
    """Return transport-independent store preparation context."""

    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommitStoreRequest:
    """Commit an engine-driven store operation."""

    key: IPCCacheServerKey
    instance_id: int
    data: bytes


@dataclass(frozen=True)
class CommitStoreResponse:
    """Report whether an engine-driven store committed successfully."""

    success: bool


@dataclass(frozen=True)
class PrepareRetrieveRequest:
    """Prepare an engine-driven retrieve operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class PrepareRetrieveResponse:
    """Return transport-independent retrieve preparation state."""

    success: bool
    data: bytes = b""
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommitRetrieveRequest:
    """Commit an engine-driven retrieve operation."""

    key: IPCCacheServerKey
    instance_id: int


@dataclass(frozen=True)
class CommitRetrieveResponse:
    """Report whether an engine-driven retrieve committed successfully."""

    success: bool


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


@dataclass(frozen=True)
class ClearRequest:
    """Clear all server caches."""


@dataclass(frozen=True)
class ClearResponse:
    """Acknowledge cache clearing."""


@dataclass(frozen=True)
class GetChunkSizeRequest:
    """Query the configured cache chunk size."""


@dataclass(frozen=True)
class GetChunkSizeResponse:
    """Return the configured cache chunk size."""

    chunk_size: int


@dataclass(frozen=True)
class GetExperimentalRequest:
    """Query enabled experimental capabilities."""


@dataclass(frozen=True)
class GetExperimentalResponse:
    """Return enabled experimental capability names."""

    names: list[str]


@dataclass(frozen=True)
class PingRequest:
    """Refresh worker liveness or probe server health."""

    instance_id: int | None


@dataclass(frozen=True)
class PingResponse:
    """Return server health."""

    ok: bool


@dataclass(frozen=True)
class NoopRequest:
    """Issue a no-op request."""


@dataclass(frozen=True)
class NoopResponse:
    """Return the no-op diagnostic message."""

    message: str


@dataclass(frozen=True)
class ReportBlockAllocationRequest:
    """Report GPU block-allocation changes."""

    instance_id: int
    model_name: str
    records: list[BlockAllocationRecord]


@dataclass(frozen=True)
class ReportBlockAllocationResponse:
    """Acknowledge a block-allocation report."""


@dataclass(frozen=True)
class CbProtocolHandshakeRequest:
    """Exchange CacheBlend protocol versions."""

    client_version: int


@dataclass(frozen=True)
class CbProtocolHandshakeResponse:
    """Return the server protocol version and compatibility result."""

    server_version: int
    client_compatible: bool


@dataclass(frozen=True)
class CbRegisterRopeRequest:
    """Register CacheBlend RoPE state."""

    instance_id: int
    cos_sin_caches_ipc: list[DeviceIPCWrapper]
    head_size: int
    is_neox_style: bool
    group_to_cache: list[int]
    group_rot: list[list[int]]


@dataclass(frozen=True)
class CbRegisterRopeResponse:
    """Acknowledge CacheBlend RoPE registration."""


@dataclass(frozen=True)
class CbUnregisterRopeRequest:
    """Unregister CacheBlend RoPE state."""

    instance_id: int


@dataclass(frozen=True)
class CbUnregisterRopeResponse:
    """Acknowledge CacheBlend RoPE removal."""


@dataclass(frozen=True)
class CbRetrievePreComputedRequest:
    """Retrieve CacheBlend pre-computed segments."""

    key: IPCCacheServerKey
    cb_match_result: list[CBMatchResult]
    gpu_block_ids: list[list[int]]
    instance_id: int
    event_ipc_handle: bytes


@dataclass(frozen=True)
class CbRetrievePreComputedResponse:
    """Return the completion event for a CacheBlend retrieve."""

    result: EventIpcHandleResult


@dataclass(frozen=True)
class CbUnifiedLookupRequest:
    """Run a unified prefix and CacheBlend lookup."""

    key: IPCCacheServerKey
    tp_size: int


@dataclass(frozen=True)
class CbUnifiedLookupResponse:
    """Return a completed unified lookup result, if available."""

    payload: CBUnifiedLookupResult | None


@dataclass(frozen=True)
class P2pLookupAndLockRequest:
    """Look up and read-lock local P2P objects."""

    keys: list[ObjectKey]
    group_layout_descs: dict[int, MemoryLayoutDesc]


@dataclass(frozen=True)
class P2pLookupAndLockResponse:
    """Return the asynchronous P2P lookup task identifier."""

    task_id: int


@dataclass(frozen=True)
class P2pQueryLookupResultsRequest:
    """Query the result of a P2P lookup task."""

    task_id: int


@dataclass(frozen=True)
class P2pQueryLookupResultsResponse:
    """Return transfer addresses when the P2P lookup has completed."""

    addresses: list[TransferChannelAddress] | None


@dataclass(frozen=True)
class P2pUnlockObjectsRequest:
    """Release P2P object read locks."""

    keys: list[ObjectKey]


@dataclass(frozen=True)
class P2pUnlockObjectsResponse:
    """Acknowledge P2P object unlocking."""


# QStore RPCs intentionally reuse RegisterKvCacheRequest,
# RegisterKvCacheResponse, UnregisterKvCacheRequest,
# UnregisterKvCacheResponse, StoreRequest, and StoreResponse because the
# protobuf service reuses the same request and response messages.


RpcRequest = (
    RegisterKvCacheRequest
    | UnregisterKvCacheRequest
    | StoreRequest
    | RetrieveRequest
    | RegisterKvCacheEngineDrivenContextRequest
    | UnregisterKvCacheEngineDrivenContextRequest
    | PrepareStoreRequest
    | CommitStoreRequest
    | PrepareRetrieveRequest
    | CommitRetrieveRequest
    | LookupRequest
    | QueryPrefetchStatusRequest
    | WaitPrefetchStatusRequest
    | QueryPrefetchLookupHitsRequest
    | FreeLookupLocksRequest
    | EndSessionRequest
    | ClearRequest
    | GetChunkSizeRequest
    | GetExperimentalRequest
    | PingRequest
    | NoopRequest
    | ReportBlockAllocationRequest
    | CbProtocolHandshakeRequest
    | CbRegisterRopeRequest
    | CbUnregisterRopeRequest
    | CbRetrievePreComputedRequest
    | CbUnifiedLookupRequest
    | P2pLookupAndLockRequest
    | P2pQueryLookupResultsRequest
    | P2pUnlockObjectsRequest
)

RpcResponse = (
    RegisterKvCacheResponse
    | UnregisterKvCacheResponse
    | StoreResponse
    | RetrieveResponse
    | RegisterKvCacheEngineDrivenContextResponse
    | UnregisterKvCacheEngineDrivenContextResponse
    | PrepareStoreResponse
    | CommitStoreResponse
    | PrepareRetrieveResponse
    | CommitRetrieveResponse
    | LookupResponse
    | QueryPrefetchStatusResponse
    | WaitPrefetchStatusResponse
    | QueryPrefetchLookupHitsResponse
    | FreeLookupLocksResponse
    | EndSessionResponse
    | ClearResponse
    | GetChunkSizeResponse
    | GetExperimentalResponse
    | PingResponse
    | NoopResponse
    | ReportBlockAllocationResponse
    | CbProtocolHandshakeResponse
    | CbRegisterRopeResponse
    | CbUnregisterRopeResponse
    | CbRetrievePreComputedResponse
    | CbUnifiedLookupResponse
    | P2pLookupAndLockResponse
    | P2pQueryLookupResultsResponse
    | P2pUnlockObjectsResponse
)


RPC_MESSAGE_TYPES: dict[str, tuple[type[RpcRequest], type[RpcResponse]]] = {
    "REGISTER_KV_CACHE": (RegisterKvCacheRequest, RegisterKvCacheResponse),
    "UNREGISTER_KV_CACHE": (UnregisterKvCacheRequest, UnregisterKvCacheResponse),
    "REGISTER_Q_CACHE": (RegisterKvCacheRequest, RegisterKvCacheResponse),
    "UNREGISTER_Q_CACHE": (UnregisterKvCacheRequest, UnregisterKvCacheResponse),
    "STORE_Q": (StoreRequest, StoreResponse),
    "STORE": (StoreRequest, StoreResponse),
    "RETRIEVE": (RetrieveRequest, RetrieveResponse),
    "LOOKUP": (LookupRequest, LookupResponse),
    "QUERY_PREFETCH_STATUS": (
        QueryPrefetchStatusRequest,
        QueryPrefetchStatusResponse,
    ),
    "WAIT_PREFETCH_STATUS": (WaitPrefetchStatusRequest, WaitPrefetchStatusResponse),
    "QUERY_PREFETCH_LOOKUP_HITS": (
        QueryPrefetchLookupHitsRequest,
        QueryPrefetchLookupHitsResponse,
    ),
    "FREE_LOOKUP_LOCKS": (FreeLookupLocksRequest, FreeLookupLocksResponse),
    "END_SESSION": (EndSessionRequest, EndSessionResponse),
    "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": (
        RegisterKvCacheEngineDrivenContextRequest,
        RegisterKvCacheEngineDrivenContextResponse,
    ),
    "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": (
        UnregisterKvCacheEngineDrivenContextRequest,
        UnregisterKvCacheEngineDrivenContextResponse,
    ),
    "PREPARE_STORE": (PrepareStoreRequest, PrepareStoreResponse),
    "COMMIT_STORE": (CommitStoreRequest, CommitStoreResponse),
    "PREPARE_RETRIEVE": (PrepareRetrieveRequest, PrepareRetrieveResponse),
    "COMMIT_RETRIEVE": (CommitRetrieveRequest, CommitRetrieveResponse),
    "CLEAR": (ClearRequest, ClearResponse),
    "GET_CHUNK_SIZE": (GetChunkSizeRequest, GetChunkSizeResponse),
    "PING": (PingRequest, PingResponse),
    "REPORT_BLOCK_ALLOCATION": (
        ReportBlockAllocationRequest,
        ReportBlockAllocationResponse,
    ),
    "NOOP": (NoopRequest, NoopResponse),
    "CB_REGISTER_ROPE": (CbRegisterRopeRequest, CbRegisterRopeResponse),
    "CB_UNREGISTER_ROPE": (CbUnregisterRopeRequest, CbUnregisterRopeResponse),
    "CB_RETRIEVE_PRE_COMPUTED": (
        CbRetrievePreComputedRequest,
        CbRetrievePreComputedResponse,
    ),
    "CB_UNIFIED_LOOKUP": (CbUnifiedLookupRequest, CbUnifiedLookupResponse),
    "P2P_LOOKUP_AND_LOCK": (P2pLookupAndLockRequest, P2pLookupAndLockResponse),
    "P2P_QUERY_LOOKUP_RESULTS": (
        P2pQueryLookupResultsRequest,
        P2pQueryLookupResultsResponse,
    ),
    "P2P_UNLOCK_OBJECTS": (P2pUnlockObjectsRequest, P2pUnlockObjectsResponse),
    "GET_EXPERIMENTAL": (GetExperimentalRequest, GetExperimentalResponse),
    "CB_PROTOCOL_HANDSHAKE": (
        CbProtocolHandshakeRequest,
        CbProtocolHandshakeResponse,
    ),
}


def make_request_message(request_name: str, *payloads: Any) -> RpcRequest:
    """Build one transport-neutral request from the compatibility call API."""
    request_class = RPC_MESSAGE_TYPES[request_name][0]
    if (
        request_class is RegisterKvCacheEngineDrivenContextRequest
        and len(payloads) == 1
        and isinstance(payloads[0], RegisterEngineDrivenContextPayload)
    ):
        payload = payloads[0]
        return request_class(
            **{name: getattr(payload, name) for name in payload.__struct_fields__}
        )
    return request_class(*payloads)


def unwrap_response_message(response: RpcResponse) -> Any:
    """Return the legacy client result represented by a Python RPC response."""
    if isinstance(
        response,
        (StoreResponse, RetrieveResponse, CbRetrievePreComputedResponse),
    ):
        return response.result.event_ipc_handle, response.result.success
    if isinstance(response, RegisterKvCacheEngineDrivenContextResponse):
        return LegacyRegisterContextResponse(response.shm_name, response.pool_size)
    if isinstance(response, PrepareStoreResponse):
        return LegacyPrepareStoreResponse(context=response.context)
    if isinstance(response, PrepareRetrieveResponse):
        return LegacyPrepareRetrieveResponse(
            success=response.success,
            data=response.data,
            context=response.context,
        )
    response_fields = fields(response)
    if not response_fields:
        return None
    values = tuple(getattr(response, item.name) for item in response_fields)
    return values[0] if len(values) == 1 else values


def serialize_rpc_message(message: MessageT, message_type: type[MessageT]) -> bytes:
    """Serialize one transport-neutral RPC message.

    Args:
        message: Python request or response instance.
        message_type: Registered concrete message type for the RPC method.

    Returns:
        MessagePack bytes shared by every request transport.

    Raises:
        TypeError: If ``message`` does not match ``message_type`` or contains an
            unsupported value.
    """
    if not isinstance(message, message_type):
        raise TypeError(
            f"expected {message_type.__name__}, got {type(message).__name__}"
        )
    return get_customized_encoder(message_type).encode(message)


def deserialize_rpc_message(payload: bytes, message_type: type[MessageT]) -> MessageT:
    """Deserialize one transport-neutral RPC message.

    Args:
        payload: MessagePack bytes received from a request transport.
        message_type: Registered concrete message type for the RPC method.

    Returns:
        A Python request or response instance.
    """
    return get_customized_decoder(message_type).decode(payload)


__all__ = [name for name in globals() if name.endswith(("Request", "Response"))] + [
    "deserialize_rpc_message",
    "serialize_rpc_message",
]
