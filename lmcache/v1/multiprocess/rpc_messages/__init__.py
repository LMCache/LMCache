# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""Python-first, transport-neutral multiprocess RPC payload contracts.

Message definitions live beside their owning business domain.  Importing this
package loads those small modules and validates their local contract
registrations once, while preserving the established public import path.
"""

# Standard
from typing import Any, TypeAlias

# First Party
from lmcache.v1.multiprocess.rpc_messages.blend import (
    CbProtocolHandshakeRequest,
    CbProtocolHandshakeResponse,
    CbRegisterRopeRequest,
    CbRegisterRopeResponse,
    CbRetrievePreComputedRequest,
    CbRetrievePreComputedResponse,
    CbUnifiedLookupRequest,
    CbUnifiedLookupResponse,
    CbUnregisterRopeRequest,
    CbUnregisterRopeResponse,
)
from lmcache.v1.multiprocess.rpc_messages.compat import (
    make_request_message,
    unwrap_response_message,
)
from lmcache.v1.multiprocess.rpc_messages.debug import NoopRequest, NoopResponse
from lmcache.v1.multiprocess.rpc_messages.engine_driven import (
    CommitRetrieveRequest,
    CommitRetrieveResponse,
    CommitStoreRequest,
    CommitStoreResponse,
    PrepareRetrieveRequest,
    PrepareRetrieveResponse,
    PrepareStoreRequest,
    PrepareStoreResponse,
    RegisterKvCacheEngineDrivenContextRequest,
    RegisterKvCacheEngineDrivenContextResponse,
    UnregisterKvCacheEngineDrivenContextRequest,
    UnregisterKvCacheEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.rpc_messages.lmcache_driven import (
    EventIpcHandleResult,
    RegisterKvCacheRequest,
    RegisterKvCacheResponse,
    RetrieveRequest,
    RetrieveResponse,
    StoreRequest,
    StoreResponse,
    UnregisterKvCacheRequest,
    UnregisterKvCacheResponse,
)
from lmcache.v1.multiprocess.rpc_messages.lookup import (
    EndSessionRequest,
    EndSessionResponse,
    FreeLookupLocksRequest,
    FreeLookupLocksResponse,
    LookupRequest,
    LookupResponse,
    QueryPrefetchLookupHitsRequest,
    QueryPrefetchLookupHitsResponse,
    QueryPrefetchStatusRequest,
    QueryPrefetchStatusResponse,
    WaitPrefetchStatusRequest,
    WaitPrefetchStatusResponse,
)
from lmcache.v1.multiprocess.rpc_messages.management import (
    ClearRequest,
    ClearResponse,
    GetChunkSizeRequest,
    GetChunkSizeResponse,
    GetExperimentalRequest,
    GetExperimentalResponse,
    PingRequest,
    PingResponse,
)
from lmcache.v1.multiprocess.rpc_messages.observability import (
    ReportBlockAllocationRequest,
    ReportBlockAllocationResponse,
)
from lmcache.v1.multiprocess.rpc_messages.p2p import (
    P2pLookupAndLockRequest,
    P2pLookupAndLockResponse,
    P2pQueryLookupResultsRequest,
    P2pQueryLookupResultsResponse,
    P2pUnlockObjectsRequest,
    P2pUnlockObjectsResponse,
)
from lmcache.v1.multiprocess.rpc_messages.registry import (
    RPC_MESSAGE_TYPES,
    get_request_message_class,
    get_response_message_class,
    validate_rpc_message_types,
)
from lmcache.v1.multiprocess.rpc_messages.serde import (
    deserialize_rpc_message,
    serialize_rpc_message,
)

# Message classes remain plain dataclasses; callers that require a common
# annotation should not need a hand-maintained union of every domain type.
RpcRequest: TypeAlias = Any
RpcResponse: TypeAlias = Any

validate_rpc_message_types()

__all__ = [
    name
    for name in globals()
    if name.endswith(("Request", "Response"))
    or name
    in {
        "EventIpcHandleResult",
        "RPC_MESSAGE_TYPES",
        "RpcRequest",
        "RpcResponse",
        "deserialize_rpc_message",
        "get_request_message_class",
        "get_response_message_class",
        "make_request_message",
        "serialize_rpc_message",
        "unwrap_response_message",
    }
]
