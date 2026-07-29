# SPDX-License-Identifier: Apache-2.0
"""LMCache mp-mode message queue, backed by gRPC.

Each ``RequestType`` maps to a distinct unary rpc method on the
``MessageQueue`` service defined in ``proto/lmcache_mq.proto`` -- the
old msgspec envelope (uid + request_type frame + payloads) is gone and
gRPC's method routing takes over.  The request/response payload bytes
themselves still carry msgspec-encoded values today, so the surrounding
handler / client business code keeps the same signatures; a follow-up
PR can promote individual rpc methods to typed proto messages without
touching this file.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
from urllib.parse import urlparse
import inspect
import pickle
import threading

# Third Party
import grpc
import msgspec
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2_grpc as _pb2_grpc_typed,
)

# Message classes come out of the protobuf descriptor pool at runtime
# and are invisible to static analysis; rebind through Any so mypy
# does not chase every attribute lookup.
lmcache_mq_pb2: Any = _pb2_typed
lmcache_mq_pb2_grpc: Any = _pb2_grpc_typed

logger = init_logger(__name__)

T = TypeVar("T")

# gRPC channel/server options. LMCache multiprocess is a loopback
# (localhost TCP or unix socket) IPC boundary carrying KV cache
# payloads that routinely exceed the 4 MiB default; disable both
# caps so registers/stores never trip on message size.
_GRPC_UNLIMITED_MSG_OPTS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
]


# ---------------------------------------------------------------------------
# Typed rpc registry (proto messages as first-class citizens).
#
# Each entry says "for this RequestType, don't touch the msgspec envelope
# -- serialize / deserialize through these two typed proto messages
# directly".  Migrating an rpc off the legacy BytesRequest / BytesResponse
# envelope is a matter of:
#
#   1. Add a real message pair to ``lmcache_mq.proto`` (see PingRequest /
#      PingResponse) and change the rpc to use them.
#   2. Add one entry to this dict wiring the RequestType to those
#      messages and the two small Python <-> proto adapters.
#
# The adapters intentionally stay next to the registry (rather than in
# the business handler) so the whole "typed-vs-legacy" decision surface
# lives in one file that grep's cheap to audit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedRpcSpec:
    """Metadata describing a typed rpc.

    Attributes:
        request_message: The generated proto message class for the
            request; instances of it hit the wire directly.
        response_message: Same for the response.
        request_to_python: Servicer-side unpack.  Turns the incoming
            proto request into the positional Python arguments the
            handler function expects.  The tuple's shape must match
            ``protocol.get_payload_classes(request_type)``.
        python_to_request: Client-side pack.  Inverse of
            ``request_to_python`` -- takes the same positional Python
            arguments and builds the proto request that hits the wire.
        python_to_response: Servicer-side pack.  Turns the handler's
            Python return value into the proto response.
        response_to_python: Client-side unpack.  Inverse of
            ``python_to_response`` -- turns the proto response back
            into the Python value the caller expects (or ``None``).
    """

    request_message: Any
    response_message: Any
    request_to_python: Callable[[Any], tuple[Any, ...]]
    python_to_request: Callable[..., Any]
    python_to_response: Callable[[Any], Any]
    response_to_python: Callable[[Any], Any]


def _ping_request_to_python(req: "lmcache_mq_pb2.PingRequest") -> tuple[Any, ...]:
    # ``-1`` on the wire is the sentinel for "untracked prober" (the
    # legacy msgspec path used ``None`` for the same case).
    instance_id: Optional[int] = None if req.instance_id == -1 else req.instance_id
    return (instance_id,)


def _ping_python_to_request(instance_id: Optional[int]) -> "lmcache_mq_pb2.PingRequest":
    wire_id = -1 if instance_id is None else instance_id
    return lmcache_mq_pb2.PingRequest(instance_id=wire_id)


def _ping_python_to_response(result: Any) -> "lmcache_mq_pb2.PingResponse":
    return lmcache_mq_pb2.PingResponse(ok=bool(result))


def _ping_response_to_python(resp: "lmcache_mq_pb2.PingResponse") -> bool:
    return bool(resp.ok)


# --- Shared: IpcCacheServerKey <-> IPCCacheServerKey -----------------
# Consumed by every rpc that carries a cache key (Lookup today; Store /
# Retrieve / FreeLookupLocks / Prepare*/Commit* / EndSession / CB-Lookup
# variants tomorrow).  Keeping the conversion in one place means the
# per-rpc adapter is a one-liner and the dataclass' ``__post_init__``
# validation (salt char set, salt length) runs exactly on the wire
# boundary regardless of who's calling.


def _ipc_key_python_to_proto(
    key: IPCCacheServerKey,
) -> "lmcache_mq_pb2.IpcCacheServerKey":
    msg = lmcache_mq_pb2.IpcCacheServerKey(
        model_name=key.model_name,
        world_size=key.world_size,
        token_ids=list(key.token_ids),
        start=key.start,
        end=key.end,
        request_id=key.request_id,
        cache_salt=key.cache_salt,
    )
    if key.worker_id is not None:
        msg.worker_id = key.worker_id
    return msg


def _ipc_key_proto_to_python(
    msg: "lmcache_mq_pb2.IpcCacheServerKey",
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=msg.model_name,
        world_size=msg.world_size,
        worker_id=msg.worker_id if msg.HasField("worker_id") else None,
        token_ids=tuple(msg.token_ids),
        start=msg.start,
        end=msg.end,
        request_id=msg.request_id,
        cache_salt=msg.cache_salt,
    )


def _lookup_request_to_python(
    req: "lmcache_mq_pb2.LookupRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.tp_size)


def _lookup_python_to_request(
    key: IPCCacheServerKey, tp_size: int
) -> "lmcache_mq_pb2.LookupRequest":
    return lmcache_mq_pb2.LookupRequest(
        key=_ipc_key_python_to_proto(key), tp_size=tp_size
    )


def _lookup_python_to_response(result: Any) -> "lmcache_mq_pb2.LookupResponse":
    # Handler returns None on the legacy path; the typed response is empty.
    del result
    return lmcache_mq_pb2.LookupResponse()


def _lookup_response_to_python(resp: "lmcache_mq_pb2.LookupResponse") -> None:
    del resp
    return None


# --- FreeLookupLocks: same shape as Lookup, different rpc name -------


def _free_lookup_locks_request_to_python(
    req: "lmcache_mq_pb2.FreeLookupLocksRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.tp_size)


def _free_lookup_locks_python_to_request(
    key: IPCCacheServerKey, tp_size: int
) -> "lmcache_mq_pb2.FreeLookupLocksRequest":
    return lmcache_mq_pb2.FreeLookupLocksRequest(
        key=_ipc_key_python_to_proto(key), tp_size=tp_size
    )


def _free_lookup_locks_python_to_response(
    result: Any,
) -> "lmcache_mq_pb2.FreeLookupLocksResponse":
    del result
    return lmcache_mq_pb2.FreeLookupLocksResponse()


def _free_lookup_locks_response_to_python(
    resp: "lmcache_mq_pb2.FreeLookupLocksResponse",
) -> None:
    del resp
    return None


# --- EndSession: single-string payload, empty response ---------------


def _end_session_request_to_python(
    req: "lmcache_mq_pb2.EndSessionRequest",
) -> tuple[Any, ...]:
    return (req.request_id,)


def _end_session_python_to_request(
    request_id: str,
) -> "lmcache_mq_pb2.EndSessionRequest":
    return lmcache_mq_pb2.EndSessionRequest(request_id=request_id)


def _end_session_python_to_response(
    result: Any,
) -> "lmcache_mq_pb2.EndSessionResponse":
    del result
    return lmcache_mq_pb2.EndSessionResponse()


def _end_session_response_to_python(
    resp: "lmcache_mq_pb2.EndSessionResponse",
) -> None:
    del resp
    return None


# --- UnregisterKvCache / UnregisterKvCacheEngineDrivenContext:
#     symmetric [int] -> None rpcs share a single pair of helpers. -----


def _instance_id_request_to_python(req: Any) -> tuple[Any, ...]:
    return (req.instance_id,)


def _make_instance_id_python_to_request(
    message_cls: Any,
) -> Callable[[int], Any]:
    def _to_request(instance_id: int) -> Any:
        return message_cls(instance_id=instance_id)

    return _to_request


def _make_empty_python_to_response(
    message_cls: Any,
) -> Callable[[Any], Any]:
    def _to_response(result: Any) -> Any:
        del result
        return message_cls()

    return _to_response


def _empty_response_to_python(resp: Any) -> None:
    del resp
    return None


# --- Query/Wait prefetch: request_id [+ timeout] -> optional int -----


def _query_prefetch_status_request_to_python(
    req: "lmcache_mq_pb2.QueryPrefetchStatusRequest",
) -> tuple[Any, ...]:
    return (req.request_id,)


def _query_prefetch_status_python_to_request(
    request_id: str,
) -> "lmcache_mq_pb2.QueryPrefetchStatusRequest":
    return lmcache_mq_pb2.QueryPrefetchStatusRequest(request_id=request_id)


def _wait_prefetch_status_request_to_python(
    req: "lmcache_mq_pb2.WaitPrefetchStatusRequest",
) -> tuple[Any, ...]:
    return (req.request_id, req.timeout)


def _wait_prefetch_status_python_to_request(
    request_id: str, timeout: float
) -> "lmcache_mq_pb2.WaitPrefetchStatusRequest":
    return lmcache_mq_pb2.WaitPrefetchStatusRequest(
        request_id=request_id, timeout=timeout
    )


def _query_prefetch_lookup_hits_request_to_python(
    req: "lmcache_mq_pb2.QueryPrefetchLookupHitsRequest",
) -> tuple[Any, ...]:
    return (req.request_id,)


def _query_prefetch_lookup_hits_python_to_request(
    request_id: str,
) -> "lmcache_mq_pb2.QueryPrefetchLookupHitsRequest":
    return lmcache_mq_pb2.QueryPrefetchLookupHitsRequest(request_id=request_id)


# ``optional int64 chunk_count = 1`` -- absent == Python ``None``.
# The three prefetch-status rpcs share this exact shape, so one pair
# of helpers plus a message-class-bound factory covers all of them.


def _make_optional_chunk_count_python_to_response(
    message_cls: Any,
) -> Callable[[Any], Any]:
    def _to_response(result: Any) -> Any:
        msg = message_cls()
        if result is not None:
            msg.chunk_count = int(result)
        return msg

    return _to_response


def _optional_chunk_count_response_to_python(resp: Any) -> Optional[int]:
    return resp.chunk_count if resp.HasField("chunk_count") else None


# --- Clear / GetChunkSize / Noop: empty-payload rpcs -----------------


def _empty_request_to_python(req: Any) -> tuple[Any, ...]:
    del req
    return ()


def _make_empty_python_to_request(
    message_cls: Any,
) -> Callable[[], Any]:
    def _to_request() -> Any:
        return message_cls()

    return _to_request


def _get_chunk_size_python_to_response(
    result: Any,
) -> "lmcache_mq_pb2.GetChunkSizeResponse":
    return lmcache_mq_pb2.GetChunkSizeResponse(chunk_size=int(result))


def _get_chunk_size_response_to_python(
    resp: "lmcache_mq_pb2.GetChunkSizeResponse",
) -> int:
    return int(resp.chunk_size)


def _noop_python_to_response(result: Any) -> "lmcache_mq_pb2.NoopResponse":
    return lmcache_mq_pb2.NoopResponse(
        message=str(result) if result is not None else ""
    )


def _noop_response_to_python(resp: "lmcache_mq_pb2.NoopResponse") -> str:
    return resp.message


# ---------------------------------------------------------------------
# Wave 3 shared helpers.
# ---------------------------------------------------------------------


def _block_id_groups_python_to_proto(
    groups: list[list[int]],
) -> list["lmcache_mq_pb2.BlockIdGroup"]:
    return [lmcache_mq_pb2.BlockIdGroup(block_ids=g) for g in groups]


def _block_id_groups_proto_to_python(
    groups: Any,
) -> list[list[int]]:
    return [list(g.block_ids) for g in groups]


def _event_result_python_to_proto(
    result: Any,
) -> "lmcache_mq_pb2.EventIpcHandleResult":
    handle, success = result
    return lmcache_mq_pb2.EventIpcHandleResult(
        event_ipc_handle=handle, success=bool(success)
    )


def _event_result_proto_to_python(
    proto: "lmcache_mq_pb2.EventIpcHandleResult",
) -> tuple[bytes, bool]:
    return (proto.event_ipc_handle, bool(proto.success))


def _match_ranges_python_to_proto(
    ranges: list[tuple[int, int]],
) -> list["lmcache_mq_pb2.MatchRange"]:
    return [lmcache_mq_pb2.MatchRange(start=s, end=e) for (s, e) in ranges]


def _match_ranges_proto_to_python(ranges: Any) -> list[tuple[int, int]]:
    return [(r.start, r.end) for r in ranges]


def _cb_match_python_to_proto(m: CBMatchResult) -> "lmcache_mq_pb2.CbMatchResult":
    return lmcache_mq_pb2.CbMatchResult(
        old_st=m.old_st,
        old_ed=m.old_ed,
        cur_st=m.cur_st,
        cur_ed=m.cur_ed,
        hash=m.hash,
    )


def _cb_match_proto_to_python(
    msg: "lmcache_mq_pb2.CbMatchResult",
) -> CBMatchResult:
    return CBMatchResult(
        old_st=msg.old_st,
        old_ed=msg.old_ed,
        cur_st=msg.cur_st,
        cur_ed=msg.cur_ed,
        hash=msg.hash,
    )


# --- Store ------------------------------------------------------------


def _store_request_to_python(
    req: "lmcache_mq_pb2.StoreRequest",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        req.instance_id,
        _block_id_groups_proto_to_python(req.gpu_block_ids),
        req.event_ipc_handle,
    )


def _store_python_to_request(
    key: IPCCacheServerKey,
    instance_id: int,
    gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes,
) -> "lmcache_mq_pb2.StoreRequest":
    return lmcache_mq_pb2.StoreRequest(
        key=_ipc_key_python_to_proto(key),
        instance_id=instance_id,
        gpu_block_ids=_block_id_groups_python_to_proto(gpu_block_ids),
        event_ipc_handle=event_ipc_handle,
    )


def _store_python_to_response(result: Any) -> "lmcache_mq_pb2.StoreResponse":
    return lmcache_mq_pb2.StoreResponse(result=_event_result_python_to_proto(result))


def _store_response_to_python(
    resp: "lmcache_mq_pb2.StoreResponse",
) -> tuple[bytes, bool]:
    return _event_result_proto_to_python(resp.result)


# --- Retrieve ---------------------------------------------------------


def _retrieve_request_to_python(
    req: "lmcache_mq_pb2.RetrieveRequest",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        req.instance_id,
        _block_id_groups_proto_to_python(req.gpu_block_ids),
        req.event_ipc_handle,
        req.skip_first_n_tokens,
    )


def _retrieve_python_to_request(
    key: IPCCacheServerKey,
    instance_id: int,
    gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes,
    skip_first_n_tokens: int,
) -> "lmcache_mq_pb2.RetrieveRequest":
    return lmcache_mq_pb2.RetrieveRequest(
        key=_ipc_key_python_to_proto(key),
        instance_id=instance_id,
        gpu_block_ids=_block_id_groups_python_to_proto(gpu_block_ids),
        event_ipc_handle=event_ipc_handle,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def _retrieve_python_to_response(
    result: Any,
) -> "lmcache_mq_pb2.RetrieveResponse":
    return lmcache_mq_pb2.RetrieveResponse(result=_event_result_python_to_proto(result))


def _retrieve_response_to_python(
    resp: "lmcache_mq_pb2.RetrieveResponse",
) -> tuple[bytes, bool]:
    return _event_result_proto_to_python(resp.result)


# --- ReportBlockAllocation -------------------------------------------


def _block_alloc_record_python_to_proto(
    r: BlockAllocationRecord,
) -> "lmcache_mq_pb2.BlockAllocationRecord":
    return lmcache_mq_pb2.BlockAllocationRecord(
        req_id=r.req_id,
        new_block_ids=r.new_block_ids,
        new_token_ids=r.new_token_ids,
    )


def _block_alloc_record_proto_to_python(
    msg: "lmcache_mq_pb2.BlockAllocationRecord",
) -> BlockAllocationRecord:
    return BlockAllocationRecord(
        req_id=msg.req_id,
        new_block_ids=list(msg.new_block_ids),
        new_token_ids=list(msg.new_token_ids),
    )


def _report_block_alloc_request_to_python(
    req: "lmcache_mq_pb2.ReportBlockAllocationRequest",
) -> tuple[Any, ...]:
    return (
        req.instance_id,
        req.model_name,
        [_block_alloc_record_proto_to_python(r) for r in req.records],
    )


def _report_block_alloc_python_to_request(
    instance_id: int,
    model_name: str,
    records: list[BlockAllocationRecord],
) -> "lmcache_mq_pb2.ReportBlockAllocationRequest":
    return lmcache_mq_pb2.ReportBlockAllocationRequest(
        instance_id=instance_id,
        model_name=model_name,
        records=[_block_alloc_record_python_to_proto(r) for r in records],
    )


# --- CB v1 lookup / store / retrieve ---------------------------------


def _cb_lookup_request_to_python(
    req: "lmcache_mq_pb2.CbLookupPreComputedRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key),)


def _cb_lookup_python_to_request(
    key: IPCCacheServerKey,
) -> "lmcache_mq_pb2.CbLookupPreComputedRequest":
    return lmcache_mq_pb2.CbLookupPreComputedRequest(key=_ipc_key_python_to_proto(key))


def _cb_lookup_python_to_response(
    result: list[tuple[int, int]],
) -> "lmcache_mq_pb2.CbLookupPreComputedResponse":
    return lmcache_mq_pb2.CbLookupPreComputedResponse(
        ranges=_match_ranges_python_to_proto(result)
    )


def _cb_lookup_response_to_python(
    resp: "lmcache_mq_pb2.CbLookupPreComputedResponse",
) -> list[tuple[int, int]]:
    return _match_ranges_proto_to_python(resp.ranges)


def _cb_store_request_to_python(req: Any) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        req.offset,
        req.instance_id,
        req.event_ipc_handle,
    )


def _make_cb_store_python_to_request(message_cls: Any) -> Callable[..., Any]:
    def _to_request(
        key: IPCCacheServerKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> Any:
        return message_cls(
            key=_ipc_key_python_to_proto(key),
            offset=offset,
            instance_id=instance_id,
            event_ipc_handle=event_ipc_handle,
        )

    return _to_request


def _make_event_result_python_to_response(
    message_cls: Any,
) -> Callable[[Any], Any]:
    def _to_response(result: Any) -> Any:
        return message_cls(result=_event_result_python_to_proto(result))

    return _to_response


def _event_result_response_to_python(resp: Any) -> tuple[bytes, bool]:
    return _event_result_proto_to_python(resp.result)


def _cb_retrieve_request_to_python(
    req: "lmcache_mq_pb2.CbRetrievePreComputedRequest",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        _match_ranges_proto_to_python(req.ranges),
        req.offset,
        req.instance_id,
        req.event_ipc_handle,
    )


def _cb_retrieve_python_to_request(
    key: IPCCacheServerKey,
    ranges: list[tuple[int, int]],
    offset: int,
    instance_id: int,
    event_ipc_handle: bytes,
) -> "lmcache_mq_pb2.CbRetrievePreComputedRequest":
    return lmcache_mq_pb2.CbRetrievePreComputedRequest(
        key=_ipc_key_python_to_proto(key),
        ranges=_match_ranges_python_to_proto(ranges),
        offset=offset,
        instance_id=instance_id,
        event_ipc_handle=event_ipc_handle,
    )


# --- CB v2: lookup returns list[CBMatchResult]; retrieve consumes it -


def _cb_lookup_v2_request_to_python(
    req: "lmcache_mq_pb2.CbLookupPreComputedV2Request",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key),)


def _cb_lookup_v2_python_to_request(
    key: IPCCacheServerKey,
) -> "lmcache_mq_pb2.CbLookupPreComputedV2Request":
    return lmcache_mq_pb2.CbLookupPreComputedV2Request(
        key=_ipc_key_python_to_proto(key)
    )


def _cb_lookup_v2_python_to_response(
    result: list[CBMatchResult],
) -> "lmcache_mq_pb2.CbLookupPreComputedV2Response":
    return lmcache_mq_pb2.CbLookupPreComputedV2Response(
        matches=[_cb_match_python_to_proto(m) for m in result]
    )


def _cb_lookup_v2_response_to_python(
    resp: "lmcache_mq_pb2.CbLookupPreComputedV2Response",
) -> list[CBMatchResult]:
    return [_cb_match_proto_to_python(m) for m in resp.matches]


def _cb_retrieve_v2_request_to_python(
    req: "lmcache_mq_pb2.CbRetrievePreComputedV2Request",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        [_cb_match_proto_to_python(m) for m in req.cb_match_result],
        req.offset,
        req.instance_id,
        req.event_ipc_handle,
    )


def _cb_retrieve_v2_python_to_request(
    key: IPCCacheServerKey,
    cb_match_result: list[CBMatchResult],
    offset: int,
    instance_id: int,
    event_ipc_handle: bytes,
) -> "lmcache_mq_pb2.CbRetrievePreComputedV2Request":
    return lmcache_mq_pb2.CbRetrievePreComputedV2Request(
        key=_ipc_key_python_to_proto(key),
        cb_match_result=[_cb_match_python_to_proto(m) for m in cb_match_result],
        offset=offset,
        instance_id=instance_id,
        event_ipc_handle=event_ipc_handle,
    )


# --- CB v3: retrieve into paged blocks; unified lookup ---------------


def _cb_retrieve_v3_request_to_python(
    req: "lmcache_mq_pb2.CbRetrievePreComputedV3Request",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        [_cb_match_proto_to_python(m) for m in req.cb_match_result],
        _block_id_groups_proto_to_python(req.gpu_block_ids),
        req.instance_id,
        req.event_ipc_handle,
    )


def _cb_retrieve_v3_python_to_request(
    key: IPCCacheServerKey,
    cb_match_result: list[CBMatchResult],
    gpu_block_ids: list[list[int]],
    instance_id: int,
    event_ipc_handle: bytes,
) -> "lmcache_mq_pb2.CbRetrievePreComputedV3Request":
    return lmcache_mq_pb2.CbRetrievePreComputedV3Request(
        key=_ipc_key_python_to_proto(key),
        cb_match_result=[_cb_match_python_to_proto(m) for m in cb_match_result],
        gpu_block_ids=_block_id_groups_python_to_proto(gpu_block_ids),
        instance_id=instance_id,
        event_ipc_handle=event_ipc_handle,
    )


def _cb_unified_lookup_request_to_python(
    req: "lmcache_mq_pb2.CbUnifiedLookupRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.tp_size)


def _cb_unified_lookup_python_to_request(
    key: IPCCacheServerKey, tp_size: int
) -> "lmcache_mq_pb2.CbUnifiedLookupRequest":
    return lmcache_mq_pb2.CbUnifiedLookupRequest(
        key=_ipc_key_python_to_proto(key), tp_size=tp_size
    )


def _cb_unified_lookup_python_to_response(
    result: Optional[CBUnifiedLookupResult],
) -> "lmcache_mq_pb2.CbUnifiedLookupResponse":
    resp = lmcache_mq_pb2.CbUnifiedLookupResponse()
    if result is not None:
        resp.payload.prefix_coverage_tokens = result.prefix_coverage_tokens
        resp.payload.non_prefix_segments.extend(
            _cb_match_python_to_proto(m) for m in result.non_prefix_segments
        )
        resp.payload.segmented_prefix_segments.extend(
            _cb_match_python_to_proto(m) for m in result.segmented_prefix_segments
        )
    return resp


def _cb_unified_lookup_response_to_python(
    resp: "lmcache_mq_pb2.CbUnifiedLookupResponse",
) -> Optional[CBUnifiedLookupResult]:
    if not resp.HasField("payload"):
        return None
    p = resp.payload
    return CBUnifiedLookupResult(
        prefix_coverage_tokens=p.prefix_coverage_tokens,
        non_prefix_segments=[
            _cb_match_proto_to_python(m) for m in p.non_prefix_segments
        ],
        segmented_prefix_segments=[
            _cb_match_proto_to_python(m) for m in p.segmented_prefix_segments
        ],
    )


# ---------------------------------------------------------------------
# Wave 4 helpers.
# ---------------------------------------------------------------------


# ``context: dict`` on the wire is pickle bytes; empty dict -> b"".
# Consolidated so PrepareStore / CommitStore / PrepareRetrieve all
# behave the same way.
def _pickle_context_to_bytes(ctx: Optional[dict]) -> bytes:
    if not ctx:
        return b""
    return pickle.dumps(ctx)


def _pickle_context_from_bytes(data: bytes) -> dict:
    if not data:
        return {}
    return pickle.loads(data)


# --- RegisterKvCacheEngineDrivenContext ------------------------------


def _register_edc_request_to_python(
    req: "lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextRequest",
) -> tuple[Any, ...]:
    return (
        RegisterEngineDrivenContextPayload(
            instance_id=req.instance_id,
            model_name=req.model_name,
            world_size=req.world_size,
            block_size=req.block_size,
            num_layers=req.num_layers,
            hidden_dim_size=req.hidden_dim_size,
            dtype_str=req.dtype_str,
            use_mla=req.use_mla,
        ),
    )


def _register_edc_python_to_request(
    payload: RegisterEngineDrivenContextPayload,
) -> "lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextRequest":
    return lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextRequest(
        instance_id=payload.instance_id,
        model_name=payload.model_name,
        world_size=payload.world_size,
        block_size=payload.block_size,
        num_layers=payload.num_layers,
        hidden_dim_size=payload.hidden_dim_size,
        dtype_str=payload.dtype_str,
        use_mla=payload.use_mla,
    )


def _register_edc_python_to_response(
    result: RegisterEngineDrivenContextResponse,
) -> "lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextResponse":
    return lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextResponse(
        shm_name=result.shm_name,
        pool_size=result.pool_size,
    )


def _register_edc_response_to_python(
    resp: "lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextResponse",
) -> RegisterEngineDrivenContextResponse:
    return RegisterEngineDrivenContextResponse(
        shm_name=resp.shm_name,
        pool_size=resp.pool_size,
    )


# --- PrepareStore / CommitStore / PrepareRetrieve / CommitRetrieve ---


def _prepare_store_request_to_python(
    req: "lmcache_mq_pb2.PrepareStoreRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.instance_id)


def _prepare_store_python_to_request(
    key: IPCCacheServerKey, instance_id: int
) -> "lmcache_mq_pb2.PrepareStoreRequest":
    return lmcache_mq_pb2.PrepareStoreRequest(
        key=_ipc_key_python_to_proto(key), instance_id=instance_id
    )


def _prepare_store_python_to_response(
    result: PrepareStoreResponse,
) -> "lmcache_mq_pb2.PrepareStoreResponse":
    return lmcache_mq_pb2.PrepareStoreResponse(
        pickled_context=_pickle_context_to_bytes(result.context)
    )


def _prepare_store_response_to_python(
    resp: "lmcache_mq_pb2.PrepareStoreResponse",
) -> PrepareStoreResponse:
    return PrepareStoreResponse(
        context=_pickle_context_from_bytes(resp.pickled_context)
    )


def _commit_store_request_to_python(
    req: "lmcache_mq_pb2.CommitStoreRequest",
) -> tuple[Any, ...]:
    return (
        _ipc_key_proto_to_python(req.key),
        req.instance_id,
        req.pickled_context,
    )


def _commit_store_python_to_request(
    key: IPCCacheServerKey, instance_id: int, context_bytes: bytes
) -> "lmcache_mq_pb2.CommitStoreRequest":
    return lmcache_mq_pb2.CommitStoreRequest(
        key=_ipc_key_python_to_proto(key),
        instance_id=instance_id,
        pickled_context=context_bytes,
    )


def _commit_store_python_to_response(
    result: bool,
) -> "lmcache_mq_pb2.CommitStoreResponse":
    return lmcache_mq_pb2.CommitStoreResponse(success=bool(result))


def _commit_store_response_to_python(
    resp: "lmcache_mq_pb2.CommitStoreResponse",
) -> bool:
    return bool(resp.success)


def _prepare_retrieve_request_to_python(
    req: "lmcache_mq_pb2.PrepareRetrieveRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.instance_id)


def _prepare_retrieve_python_to_request(
    key: IPCCacheServerKey, instance_id: int
) -> "lmcache_mq_pb2.PrepareRetrieveRequest":
    return lmcache_mq_pb2.PrepareRetrieveRequest(
        key=_ipc_key_python_to_proto(key), instance_id=instance_id
    )


def _prepare_retrieve_python_to_response(
    result: PrepareRetrieveResponse,
) -> "lmcache_mq_pb2.PrepareRetrieveResponse":
    return lmcache_mq_pb2.PrepareRetrieveResponse(
        success=bool(result.success),
        data=result.data,
        pickled_context=_pickle_context_to_bytes(result.context),
    )


def _prepare_retrieve_response_to_python(
    resp: "lmcache_mq_pb2.PrepareRetrieveResponse",
) -> PrepareRetrieveResponse:
    return PrepareRetrieveResponse(
        success=bool(resp.success),
        data=resp.data,
        context=_pickle_context_from_bytes(resp.pickled_context),
    )


def _commit_retrieve_request_to_python(
    req: "lmcache_mq_pb2.CommitRetrieveRequest",
) -> tuple[Any, ...]:
    return (_ipc_key_proto_to_python(req.key), req.instance_id)


def _commit_retrieve_python_to_request(
    key: IPCCacheServerKey, instance_id: int
) -> "lmcache_mq_pb2.CommitRetrieveRequest":
    return lmcache_mq_pb2.CommitRetrieveRequest(
        key=_ipc_key_python_to_proto(key), instance_id=instance_id
    )


def _commit_retrieve_python_to_response(
    result: bool,
) -> "lmcache_mq_pb2.CommitRetrieveResponse":
    return lmcache_mq_pb2.CommitRetrieveResponse(success=bool(result))


def _commit_retrieve_response_to_python(
    resp: "lmcache_mq_pb2.CommitRetrieveResponse",
) -> bool:
    return bool(resp.success)


# --- P2P shared helpers ----------------------------------------------


def _object_key_python_to_proto(k: ObjectKey) -> "lmcache_mq_pb2.ObjectKey":
    return lmcache_mq_pb2.ObjectKey(
        chunk_hash=k.chunk_hash,
        model_name=k.model_name,
        kv_rank=k.kv_rank,
        object_group_id=k.object_group_id,
        cache_salt=k.cache_salt,
    )


def _object_key_proto_to_python(msg: "lmcache_mq_pb2.ObjectKey") -> ObjectKey:
    return ObjectKey(
        chunk_hash=msg.chunk_hash,
        model_name=msg.model_name,
        kv_rank=msg.kv_rank,
        object_group_id=msg.object_group_id,
        cache_salt=msg.cache_salt,
    )


def _dtype_to_wire(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_from_wire(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError("unknown torch dtype name: " + repr(name))
    return dt


def _layout_python_to_proto(
    layout: MemoryLayoutDesc,
) -> "lmcache_mq_pb2.MemoryLayoutDesc":
    return lmcache_mq_pb2.MemoryLayoutDesc(
        shapes=[lmcache_mq_pb2.TensorShape(dims=list(s)) for s in layout.shapes],
        dtypes=[_dtype_to_wire(dt) for dt in layout.dtypes],
    )


def _layout_proto_to_python(
    msg: "lmcache_mq_pb2.MemoryLayoutDesc",
) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size(list(s.dims)) for s in msg.shapes],
        dtypes=[_dtype_from_wire(n) for n in msg.dtypes],
    )


def _addr_python_to_proto(
    a: TransferChannelAddress,
) -> "lmcache_mq_pb2.TransferChannelAddress":
    return lmcache_mq_pb2.TransferChannelAddress(offset=a.offset, size=a.size)


def _addr_proto_to_python(
    msg: "lmcache_mq_pb2.TransferChannelAddress",
) -> TransferChannelAddress:
    return TransferChannelAddress(offset=msg.offset, size=msg.size)


# --- P2P_LOOKUP_AND_LOCK / QUERY / UNLOCK ----------------------------


def _p2p_lookup_request_to_python(
    req: "lmcache_mq_pb2.P2pLookupAndLockRequest",
) -> tuple[Any, ...]:
    return (
        [_object_key_proto_to_python(k) for k in req.keys],
        _layout_proto_to_python(req.layout_desc),
    )


def _p2p_lookup_python_to_request(
    keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
) -> "lmcache_mq_pb2.P2pLookupAndLockRequest":
    return lmcache_mq_pb2.P2pLookupAndLockRequest(
        keys=[_object_key_python_to_proto(k) for k in keys],
        layout_desc=_layout_python_to_proto(layout_desc),
    )


def _p2p_lookup_python_to_response(
    result: int,
) -> "lmcache_mq_pb2.P2pLookupAndLockResponse":
    return lmcache_mq_pb2.P2pLookupAndLockResponse(task_id=int(result))


def _p2p_lookup_response_to_python(
    resp: "lmcache_mq_pb2.P2pLookupAndLockResponse",
) -> int:
    return int(resp.task_id)


def _p2p_query_request_to_python(
    req: "lmcache_mq_pb2.P2pQueryLookupResultsRequest",
) -> tuple[Any, ...]:
    return (req.task_id,)


def _p2p_query_python_to_request(
    task_id: int,
) -> "lmcache_mq_pb2.P2pQueryLookupResultsRequest":
    return lmcache_mq_pb2.P2pQueryLookupResultsRequest(task_id=task_id)


def _p2p_query_python_to_response(
    result: Optional[list[TransferChannelAddress]],
) -> "lmcache_mq_pb2.P2pQueryLookupResultsResponse":
    resp = lmcache_mq_pb2.P2pQueryLookupResultsResponse()
    if result is not None:
        resp.addresses.addresses.extend(_addr_python_to_proto(a) for a in result)
    return resp


def _p2p_query_response_to_python(
    resp: "lmcache_mq_pb2.P2pQueryLookupResultsResponse",
) -> Optional[list[TransferChannelAddress]]:
    if not resp.HasField("addresses"):
        return None
    return [_addr_proto_to_python(a) for a in resp.addresses.addresses]


def _p2p_unlock_request_to_python(
    req: "lmcache_mq_pb2.P2pUnlockObjectsRequest",
) -> tuple[Any, ...]:
    return ([_object_key_proto_to_python(k) for k in req.keys],)


def _p2p_unlock_python_to_request(
    keys: list[ObjectKey],
) -> "lmcache_mq_pb2.P2pUnlockObjectsRequest":
    return lmcache_mq_pb2.P2pUnlockObjectsRequest(
        keys=[_object_key_python_to_proto(k) for k in keys]
    )


# ---------------------------------------------------------------------
# Wave 5 helpers.
# ---------------------------------------------------------------------


# DeviceIPCWrapper preserves its concrete subclass via pickle -- the
# wrapper base class ships Serialize/Deserialize static methods for
# exactly this reason.  Reusing them keeps the wire format aligned
# with the on-disk / cross-process behaviour without introducing a
# second serialization path.
def _wrapper_python_to_proto(
    w: DeviceIPCWrapper,
) -> "lmcache_mq_pb2.DeviceIpcWrapper":
    return lmcache_mq_pb2.DeviceIpcWrapper(
        pickled_payload=DeviceIPCWrapper.Serialize(w)
    )


def _wrapper_proto_to_python(
    msg: "lmcache_mq_pb2.DeviceIpcWrapper",
) -> DeviceIPCWrapper:
    return DeviceIPCWrapper.Deserialize(msg.pickled_payload)


def _wrappers_python_to_proto(
    wrappers: list[DeviceIPCWrapper],
) -> list["lmcache_mq_pb2.DeviceIpcWrapper"]:
    return [_wrapper_python_to_proto(w) for w in wrappers]


def _wrappers_proto_to_python(msgs: Any) -> list[DeviceIPCWrapper]:
    return [_wrapper_proto_to_python(m) for m in msgs]


def _engine_group_info_python_to_proto(
    info: EngineGroupInfo,
) -> "lmcache_mq_pb2.EngineGroupInfo":
    return lmcache_mq_pb2.EngineGroupInfo(
        engine_group_id=info.engine_group_id,
        layer_indices=list(info.layer_indices),
        tokens_per_block=info.tokens_per_block,
        sw_size_tokens=info.sw_size_tokens,
    )


def _engine_group_info_proto_to_python(
    msg: "lmcache_mq_pb2.EngineGroupInfo",
) -> EngineGroupInfo:
    return EngineGroupInfo(
        engine_group_id=msg.engine_group_id,
        layer_indices=tuple(msg.layer_indices),
        tokens_per_block=msg.tokens_per_block,
        sw_size_tokens=msg.sw_size_tokens,
    )


# LayoutHints is a TypedDict; empty dict maps to empty bytes on the
# wire so a fresh caller talking to an older peer still round-trips.
def _layout_hints_to_bytes(hints: Optional[LayoutHints]) -> bytes:
    if not hints:
        return b""
    return pickle.dumps(dict(hints))


def _layout_hints_from_bytes(data: bytes) -> LayoutHints:
    if not data:
        return {}
    return pickle.loads(data)


# --- RegisterKvCache -------------------------------------------------


def _register_kv_cache_request_to_python(
    req: "lmcache_mq_pb2.RegisterKvCacheRequest",
) -> tuple[Any, ...]:
    return (
        req.instance_id,
        _wrappers_proto_to_python(req.kv_cache),
        req.model_name,
        req.world_size,
        EngineType(req.engine_type),
        _layout_hints_from_bytes(req.pickled_layout_hints),
        [_engine_group_info_proto_to_python(g) for g in req.engine_group_infos],
    )


def _register_kv_cache_python_to_request(
    instance_id: int,
    kv_cache: list[DeviceIPCWrapper],
    model_name: str,
    world_size: int,
    engine_type: EngineType,
    layout_hints: Optional[LayoutHints],
    engine_group_infos: list[EngineGroupInfo],
) -> "lmcache_mq_pb2.RegisterKvCacheRequest":
    return lmcache_mq_pb2.RegisterKvCacheRequest(
        instance_id=instance_id,
        kv_cache=_wrappers_python_to_proto(kv_cache),
        model_name=model_name,
        world_size=world_size,
        engine_type=engine_type.value,
        pickled_layout_hints=_layout_hints_to_bytes(layout_hints),
        engine_group_infos=[
            _engine_group_info_python_to_proto(g) for g in engine_group_infos
        ],
    )


# --- CbRegisterKvCache ----------------------------------------------


def _cb_register_kv_cache_request_to_python(
    req: "lmcache_mq_pb2.CbRegisterKvCacheRequest",
) -> tuple[Any, ...]:
    return (
        req.instance_id,
        _wrappers_proto_to_python(req.kv_cache),
        req.model_name,
        req.world_size,
    )


def _cb_register_kv_cache_python_to_request(
    instance_id: int,
    kv_cache: list[DeviceIPCWrapper],
    model_name: str,
    world_size: int,
) -> "lmcache_mq_pb2.CbRegisterKvCacheRequest":
    return lmcache_mq_pb2.CbRegisterKvCacheRequest(
        instance_id=instance_id,
        kv_cache=_wrappers_python_to_proto(kv_cache),
        model_name=model_name,
        world_size=world_size,
    )


# --- CbRegisterRopeV3 -----------------------------------------------


def _cb_register_rope_v3_request_to_python(
    req: "lmcache_mq_pb2.CbRegisterRopeV3Request",
) -> tuple[Any, ...]:
    return (
        req.instance_id,
        _wrappers_proto_to_python(req.cos_sin_caches_ipc),
        req.head_size,
        bool(req.is_neox_style),
        list(req.group_to_cache),
    )


def _cb_register_rope_v3_python_to_request(
    instance_id: int,
    cos_sin_caches_ipc: list[DeviceIPCWrapper],
    head_size: int,
    is_neox_style: bool,
    group_to_cache: list[int],
) -> "lmcache_mq_pb2.CbRegisterRopeV3Request":
    return lmcache_mq_pb2.CbRegisterRopeV3Request(
        instance_id=instance_id,
        cos_sin_caches_ipc=_wrappers_python_to_proto(cos_sin_caches_ipc),
        head_size=head_size,
        is_neox_style=bool(is_neox_style),
        group_to_cache=list(group_to_cache),
    )


# ---------------------------------------------------------------------------
# msgspec encode / decode helpers (payload bytes wrapped inside proto)
# ---------------------------------------------------------------------------

_SPECIAL_ENCODER_DECODERS = {
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    MemoryLayoutDesc: (
        get_customized_encoder(MemoryLayoutDesc),
        get_customized_decoder(MemoryLayoutDesc),
    ),
    dict[int, MemoryLayoutDesc]: (
        get_customized_encoder(dict[int, MemoryLayoutDesc]),
        get_customized_decoder(dict[int, MemoryLayoutDesc]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


def unwrap_request_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> list[Any]:
    if len(b_payloads) != len(payload_clss):
        raise ValueError("Payload count does not match expected count")

    return [
        msgspec_decode(payload, cls=cls)
        for payload, cls in zip(b_payloads, payload_clss, strict=False)
    ]


# The one source of truth for "which RequestType has been promoted to a
# typed proto message pair".  Entries here take priority over the
# msgspec-envelope path in both ``submit_request`` and the servicer.
_TYPED_RPCS: dict[RequestType, TypedRpcSpec] = {
    RequestType.PING: TypedRpcSpec(
        request_message=lmcache_mq_pb2.PingRequest,
        response_message=lmcache_mq_pb2.PingResponse,
        request_to_python=_ping_request_to_python,
        python_to_request=_ping_python_to_request,
        python_to_response=_ping_python_to_response,
        response_to_python=_ping_response_to_python,
    ),
    RequestType.LOOKUP: TypedRpcSpec(
        request_message=lmcache_mq_pb2.LookupRequest,
        response_message=lmcache_mq_pb2.LookupResponse,
        request_to_python=_lookup_request_to_python,
        python_to_request=_lookup_python_to_request,
        python_to_response=_lookup_python_to_response,
        response_to_python=_lookup_response_to_python,
    ),
    RequestType.FREE_LOOKUP_LOCKS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.FreeLookupLocksRequest,
        response_message=lmcache_mq_pb2.FreeLookupLocksResponse,
        request_to_python=_free_lookup_locks_request_to_python,
        python_to_request=_free_lookup_locks_python_to_request,
        python_to_response=_free_lookup_locks_python_to_response,
        response_to_python=_free_lookup_locks_response_to_python,
    ),
    RequestType.END_SESSION: TypedRpcSpec(
        request_message=lmcache_mq_pb2.EndSessionRequest,
        response_message=lmcache_mq_pb2.EndSessionResponse,
        request_to_python=_end_session_request_to_python,
        python_to_request=_end_session_python_to_request,
        python_to_response=_end_session_python_to_response,
        response_to_python=_end_session_response_to_python,
    ),
    RequestType.UNREGISTER_KV_CACHE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.UnregisterKvCacheRequest,
        response_message=lmcache_mq_pb2.UnregisterKvCacheResponse,
        request_to_python=_instance_id_request_to_python,
        python_to_request=_make_instance_id_python_to_request(
            lmcache_mq_pb2.UnregisterKvCacheRequest
        ),
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.UnregisterKvCacheResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT: TypedRpcSpec(
        request_message=lmcache_mq_pb2.UnregisterKvCacheEngineDrivenContextRequest,
        response_message=lmcache_mq_pb2.UnregisterKvCacheEngineDrivenContextResponse,
        request_to_python=_instance_id_request_to_python,
        python_to_request=_make_instance_id_python_to_request(
            lmcache_mq_pb2.UnregisterKvCacheEngineDrivenContextRequest
        ),
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.UnregisterKvCacheEngineDrivenContextResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.QUERY_PREFETCH_STATUS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.QueryPrefetchStatusRequest,
        response_message=lmcache_mq_pb2.QueryPrefetchStatusResponse,
        request_to_python=_query_prefetch_status_request_to_python,
        python_to_request=_query_prefetch_status_python_to_request,
        python_to_response=_make_optional_chunk_count_python_to_response(
            lmcache_mq_pb2.QueryPrefetchStatusResponse
        ),
        response_to_python=_optional_chunk_count_response_to_python,
    ),
    RequestType.WAIT_PREFETCH_STATUS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.WaitPrefetchStatusRequest,
        response_message=lmcache_mq_pb2.WaitPrefetchStatusResponse,
        request_to_python=_wait_prefetch_status_request_to_python,
        python_to_request=_wait_prefetch_status_python_to_request,
        python_to_response=_make_optional_chunk_count_python_to_response(
            lmcache_mq_pb2.WaitPrefetchStatusResponse
        ),
        response_to_python=_optional_chunk_count_response_to_python,
    ),
    RequestType.QUERY_PREFETCH_LOOKUP_HITS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.QueryPrefetchLookupHitsRequest,
        response_message=lmcache_mq_pb2.QueryPrefetchLookupHitsResponse,
        request_to_python=_query_prefetch_lookup_hits_request_to_python,
        python_to_request=_query_prefetch_lookup_hits_python_to_request,
        python_to_response=_make_optional_chunk_count_python_to_response(
            lmcache_mq_pb2.QueryPrefetchLookupHitsResponse
        ),
        response_to_python=_optional_chunk_count_response_to_python,
    ),
    RequestType.CLEAR: TypedRpcSpec(
        request_message=lmcache_mq_pb2.ClearRequest,
        response_message=lmcache_mq_pb2.ClearResponse,
        request_to_python=_empty_request_to_python,
        python_to_request=_make_empty_python_to_request(lmcache_mq_pb2.ClearRequest),
        python_to_response=_make_empty_python_to_response(lmcache_mq_pb2.ClearResponse),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.GET_CHUNK_SIZE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.GetChunkSizeRequest,
        response_message=lmcache_mq_pb2.GetChunkSizeResponse,
        request_to_python=_empty_request_to_python,
        python_to_request=_make_empty_python_to_request(
            lmcache_mq_pb2.GetChunkSizeRequest
        ),
        python_to_response=_get_chunk_size_python_to_response,
        response_to_python=_get_chunk_size_response_to_python,
    ),
    RequestType.NOOP: TypedRpcSpec(
        request_message=lmcache_mq_pb2.NoopRequest,
        response_message=lmcache_mq_pb2.NoopResponse,
        request_to_python=_empty_request_to_python,
        python_to_request=_make_empty_python_to_request(lmcache_mq_pb2.NoopRequest),
        python_to_response=_noop_python_to_response,
        response_to_python=_noop_response_to_python,
    ),
    RequestType.STORE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.StoreRequest,
        response_message=lmcache_mq_pb2.StoreResponse,
        request_to_python=_store_request_to_python,
        python_to_request=_store_python_to_request,
        python_to_response=_store_python_to_response,
        response_to_python=_store_response_to_python,
    ),
    RequestType.RETRIEVE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.RetrieveRequest,
        response_message=lmcache_mq_pb2.RetrieveResponse,
        request_to_python=_retrieve_request_to_python,
        python_to_request=_retrieve_python_to_request,
        python_to_response=_retrieve_python_to_response,
        response_to_python=_retrieve_response_to_python,
    ),
    RequestType.REPORT_BLOCK_ALLOCATION: TypedRpcSpec(
        request_message=lmcache_mq_pb2.ReportBlockAllocationRequest,
        response_message=lmcache_mq_pb2.ReportBlockAllocationResponse,
        request_to_python=_report_block_alloc_request_to_python,
        python_to_request=_report_block_alloc_python_to_request,
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.ReportBlockAllocationResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.CB_UNREGISTER_KV_CACHE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbUnregisterKvCacheRequest,
        response_message=lmcache_mq_pb2.CbUnregisterKvCacheResponse,
        request_to_python=_instance_id_request_to_python,
        python_to_request=_make_instance_id_python_to_request(
            lmcache_mq_pb2.CbUnregisterKvCacheRequest
        ),
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.CbUnregisterKvCacheResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.CB_UNREGISTER_ROPE_V3: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbUnregisterRopeV3Request,
        response_message=lmcache_mq_pb2.CbUnregisterRopeV3Response,
        request_to_python=_instance_id_request_to_python,
        python_to_request=_make_instance_id_python_to_request(
            lmcache_mq_pb2.CbUnregisterRopeV3Request
        ),
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.CbUnregisterRopeV3Response
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.CB_LOOKUP_PRE_COMPUTED: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbLookupPreComputedRequest,
        response_message=lmcache_mq_pb2.CbLookupPreComputedResponse,
        request_to_python=_cb_lookup_request_to_python,
        python_to_request=_cb_lookup_python_to_request,
        python_to_response=_cb_lookup_python_to_response,
        response_to_python=_cb_lookup_response_to_python,
    ),
    RequestType.CB_STORE_PRE_COMPUTED: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbStorePreComputedRequest,
        response_message=lmcache_mq_pb2.CbStorePreComputedResponse,
        request_to_python=_cb_store_request_to_python,
        python_to_request=_make_cb_store_python_to_request(
            lmcache_mq_pb2.CbStorePreComputedRequest
        ),
        python_to_response=_make_event_result_python_to_response(
            lmcache_mq_pb2.CbStorePreComputedResponse
        ),
        response_to_python=_event_result_response_to_python,
    ),
    RequestType.CB_STORE_FINAL: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbStoreFinalRequest,
        response_message=lmcache_mq_pb2.CbStoreFinalResponse,
        request_to_python=_cb_store_request_to_python,
        python_to_request=_make_cb_store_python_to_request(
            lmcache_mq_pb2.CbStoreFinalRequest
        ),
        python_to_response=_make_event_result_python_to_response(
            lmcache_mq_pb2.CbStoreFinalResponse
        ),
        response_to_python=_event_result_response_to_python,
    ),
    RequestType.CB_RETRIEVE_PRE_COMPUTED: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbRetrievePreComputedRequest,
        response_message=lmcache_mq_pb2.CbRetrievePreComputedResponse,
        request_to_python=_cb_retrieve_request_to_python,
        python_to_request=_cb_retrieve_python_to_request,
        python_to_response=_make_event_result_python_to_response(
            lmcache_mq_pb2.CbRetrievePreComputedResponse
        ),
        response_to_python=_event_result_response_to_python,
    ),
    RequestType.CB_LOOKUP_PRE_COMPUTED_V2: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbLookupPreComputedV2Request,
        response_message=lmcache_mq_pb2.CbLookupPreComputedV2Response,
        request_to_python=_cb_lookup_v2_request_to_python,
        python_to_request=_cb_lookup_v2_python_to_request,
        python_to_response=_cb_lookup_v2_python_to_response,
        response_to_python=_cb_lookup_v2_response_to_python,
    ),
    RequestType.CB_RETRIEVE_PRE_COMPUTED_V2: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbRetrievePreComputedV2Request,
        response_message=lmcache_mq_pb2.CbRetrievePreComputedV2Response,
        request_to_python=_cb_retrieve_v2_request_to_python,
        python_to_request=_cb_retrieve_v2_python_to_request,
        python_to_response=_make_event_result_python_to_response(
            lmcache_mq_pb2.CbRetrievePreComputedV2Response
        ),
        response_to_python=_event_result_response_to_python,
    ),
    RequestType.CB_RETRIEVE_PRE_COMPUTED_V3: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbRetrievePreComputedV3Request,
        response_message=lmcache_mq_pb2.CbRetrievePreComputedV3Response,
        request_to_python=_cb_retrieve_v3_request_to_python,
        python_to_request=_cb_retrieve_v3_python_to_request,
        python_to_response=_make_event_result_python_to_response(
            lmcache_mq_pb2.CbRetrievePreComputedV3Response
        ),
        response_to_python=_event_result_response_to_python,
    ),
    RequestType.CB_UNIFIED_LOOKUP: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbUnifiedLookupRequest,
        response_message=lmcache_mq_pb2.CbUnifiedLookupResponse,
        request_to_python=_cb_unified_lookup_request_to_python,
        python_to_request=_cb_unified_lookup_python_to_request,
        python_to_response=_cb_unified_lookup_python_to_response,
        response_to_python=_cb_unified_lookup_response_to_python,
    ),
    RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT: TypedRpcSpec(
        request_message=(lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextRequest),
        response_message=(lmcache_mq_pb2.RegisterKvCacheEngineDrivenContextResponse),
        request_to_python=_register_edc_request_to_python,
        python_to_request=_register_edc_python_to_request,
        python_to_response=_register_edc_python_to_response,
        response_to_python=_register_edc_response_to_python,
    ),
    RequestType.PREPARE_STORE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.PrepareStoreRequest,
        response_message=lmcache_mq_pb2.PrepareStoreResponse,
        request_to_python=_prepare_store_request_to_python,
        python_to_request=_prepare_store_python_to_request,
        python_to_response=_prepare_store_python_to_response,
        response_to_python=_prepare_store_response_to_python,
    ),
    RequestType.COMMIT_STORE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CommitStoreRequest,
        response_message=lmcache_mq_pb2.CommitStoreResponse,
        request_to_python=_commit_store_request_to_python,
        python_to_request=_commit_store_python_to_request,
        python_to_response=_commit_store_python_to_response,
        response_to_python=_commit_store_response_to_python,
    ),
    RequestType.PREPARE_RETRIEVE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.PrepareRetrieveRequest,
        response_message=lmcache_mq_pb2.PrepareRetrieveResponse,
        request_to_python=_prepare_retrieve_request_to_python,
        python_to_request=_prepare_retrieve_python_to_request,
        python_to_response=_prepare_retrieve_python_to_response,
        response_to_python=_prepare_retrieve_response_to_python,
    ),
    RequestType.COMMIT_RETRIEVE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CommitRetrieveRequest,
        response_message=lmcache_mq_pb2.CommitRetrieveResponse,
        request_to_python=_commit_retrieve_request_to_python,
        python_to_request=_commit_retrieve_python_to_request,
        python_to_response=_commit_retrieve_python_to_response,
        response_to_python=_commit_retrieve_response_to_python,
    ),
    RequestType.P2P_LOOKUP_AND_LOCK: TypedRpcSpec(
        request_message=lmcache_mq_pb2.P2pLookupAndLockRequest,
        response_message=lmcache_mq_pb2.P2pLookupAndLockResponse,
        request_to_python=_p2p_lookup_request_to_python,
        python_to_request=_p2p_lookup_python_to_request,
        python_to_response=_p2p_lookup_python_to_response,
        response_to_python=_p2p_lookup_response_to_python,
    ),
    RequestType.P2P_QUERY_LOOKUP_RESULTS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.P2pQueryLookupResultsRequest,
        response_message=lmcache_mq_pb2.P2pQueryLookupResultsResponse,
        request_to_python=_p2p_query_request_to_python,
        python_to_request=_p2p_query_python_to_request,
        python_to_response=_p2p_query_python_to_response,
        response_to_python=_p2p_query_response_to_python,
    ),
    RequestType.P2P_UNLOCK_OBJECTS: TypedRpcSpec(
        request_message=lmcache_mq_pb2.P2pUnlockObjectsRequest,
        response_message=lmcache_mq_pb2.P2pUnlockObjectsResponse,
        request_to_python=_p2p_unlock_request_to_python,
        python_to_request=_p2p_unlock_python_to_request,
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.P2pUnlockObjectsResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.REGISTER_KV_CACHE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.RegisterKvCacheRequest,
        response_message=lmcache_mq_pb2.RegisterKvCacheResponse,
        request_to_python=_register_kv_cache_request_to_python,
        python_to_request=_register_kv_cache_python_to_request,
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.RegisterKvCacheResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.CB_REGISTER_KV_CACHE: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbRegisterKvCacheRequest,
        response_message=lmcache_mq_pb2.CbRegisterKvCacheResponse,
        request_to_python=_cb_register_kv_cache_request_to_python,
        python_to_request=_cb_register_kv_cache_python_to_request,
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.CbRegisterKvCacheResponse
        ),
        response_to_python=_empty_response_to_python,
    ),
    RequestType.CB_REGISTER_ROPE_V3: TypedRpcSpec(
        request_message=lmcache_mq_pb2.CbRegisterRopeV3Request,
        response_message=lmcache_mq_pb2.CbRegisterRopeV3Response,
        request_to_python=_cb_register_rope_v3_request_to_python,
        python_to_request=_cb_register_rope_v3_python_to_request,
        python_to_response=_make_empty_python_to_response(
            lmcache_mq_pb2.CbRegisterRopeV3Response
        ),
        response_to_python=_empty_response_to_python,
    ),
}


# ---------------------------------------------------------------------------
# RequestType <-> gRPC method name
# ---------------------------------------------------------------------------


def request_type_to_method_name(request_type: RequestType) -> str:
    """Return the CamelCase gRPC method name for a ``RequestType``.

    ``STORE`` -> ``Store``; ``CB_LOOKUP_PRE_COMPUTED_V2`` ->
    ``CbLookupPreComputedV2``; ``P2P_LOOKUP_AND_LOCK`` ->
    ``P2PLookupAndLock``.  These names are baked into ``lmcache_mq.proto``
    so any drift shows up immediately at handshake time.
    """
    parts = request_type.name.split("_")
    out: list[str] = []
    for part in parts:
        if part == "P2P":
            out.append("P2P")
        else:
            out.append(part[:1].upper() + part[1:].lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def _parse_grpc_url(url: str) -> str:
    """Return a ``host:port`` target that ``grpc.insecure_channel`` accepts.

    Accepts ``grpc://host:port`` or a bare ``host:port``.  Any other
    transport scheme (``tcp://`` / ``ipc://`` / etc.) is rejected up front
    now that gRPC is the only supported transport.
    """
    if "://" not in url:
        return url
    parsed = urlparse(url)
    if parsed.scheme != "grpc":
        raise ValueError(
            f"unsupported transport scheme {parsed.scheme!r} for url {url!r}; "
            f"only grpc:// (or a bare host:port) is supported"
        )
    if not parsed.netloc:
        raise ValueError(f"missing host in url {url!r}")
    return parsed.netloc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MessageQueueClient:
    """gRPC-backed client for the LMCache mp cache server.

    Instances are cheap; a shared ``grpc.Channel`` is created per client
    and callers can share one client across many threads (gRPC channels
    are thread-safe).

    Args:
        server_url: Either ``grpc://host:port`` or a bare ``host:port``.
        context: Legacy positional slot kept for backwards compatibility
            with the historical zmq-based constructor; ignored.
    """

    def __init__(
        self,
        server_url: str,
        context: Optional[Any] = None,
        transport: Optional[Any] = None,
    ):
        del context, transport  # legacy positional slots, no longer used
        target = _parse_grpc_url(server_url)
        self._server_url = server_url
        self._channel = grpc.insecure_channel(target, options=_GRPC_UNLIMITED_MSG_OPTS)
        self._stub = lmcache_mq_pb2_grpc.MessageQueueStub(self._channel)

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
        response_cls: Optional[T] = None,
    ) -> MessagingFuture[T]:
        """Submit a request and return a future for its response.

        Args:
            request_type: Which RPC to invoke.
            request_payloads: Positional payloads matching
                ``get_payload_classes(request_type)``.
            response_cls: Kept for signature compatibility; ignored
                (the response class is resolved from ``request_type``).

        Returns:
            A ``MessagingFuture`` completed by the gRPC callback.
        """
        del response_cls
        method_name = request_type_to_method_name(request_type)
        stub_method = getattr(self._stub, method_name)
        future: MessagingFuture[T] = MessagingFuture()

        typed_spec = _TYPED_RPCS[request_type]
        proto_request = typed_spec.python_to_request(*request_payloads)

        def _on_done_typed(call: "grpc.Future[Any]") -> None:
            try:
                proto_response = call.result()
            except grpc.RpcError as exc:
                logger.error("gRPC call %s failed: %s", method_name, exc)
                future.set_result(None)  # type: ignore[arg-type]
                return
            except Exception:  # defensive
                logger.exception("gRPC call %s failed", method_name)
                future.set_result(None)  # type: ignore[arg-type]
                return
            try:
                decoded = typed_spec.response_to_python(proto_response)
            except Exception:
                logger.exception("failed to decode typed response for %s", method_name)
                future.set_result(None)  # type: ignore[arg-type]
                return
            future.set_result(decoded)

        call = stub_method.future(proto_request)
        call.add_done_callback(_on_done_typed)
        return future

    def close(self) -> None:
        self._channel.close()


# ---------------------------------------------------------------------------
# Server: RequestHandlerBase + concrete handler types (unchanged interface)
# ---------------------------------------------------------------------------


ResponseType = TypeVar("ResponseType", covariant=True)
StateType = TypeVar("StateType", covariant=True)


class RequestHandlerBase(Generic[ResponseType]):
    def __call__(self, payloads: list[bytes]):
        raise NotImplementedError

    def get_response_class(self) -> ResponseType:
        raise NotImplementedError

    def get_handler_type(self) -> HandlerType:
        raise NotImplementedError


class SyncRequestHandler(RequestHandlerBase[ResponseType]):
    """Handler that runs in the calling grpc worker thread."""

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.payload_clss = payload_clss
        self.response_cls = response_cls
        self.handler = handler

    def __call__(self, payloads: list[bytes]) -> ResponseType:
        return self.handler(*unwrap_request_payloads(payloads, self.payload_clss))

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.SYNC


class BlockingRequestHandler(RequestHandlerBase[ResponseType]):
    """Handler dispatched to a dedicated thread pool (normal or affinity)."""

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.executor: ThreadPoolExecutor | AffinityThreadPool | None = None
        self.payload_clss = payload_clss
        self.handler = handler
        self.response_cls = response_cls

    def __call__(
        self, payloads: list[bytes], affinity_key: Any = 0
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "BlockingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler, *decoded_payloads, affinity_key=affinity_key
            )
        return self.executor.submit(self.handler, *decoded_payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.BLOCKING


class NonBlockingRequestHandler(Generic[ResponseType, StateType]):
    """Reserved for future async handlers; not currently instantiated."""

    pass


# ---------------------------------------------------------------------------
# Server: gRPC servicer bridging RequestType -> RequestHandlerBase
# ---------------------------------------------------------------------------


class _RequestHandlerServicer(lmcache_mq_pb2_grpc.MessageQueueServicer):
    """Bridge every rpc method to the ``RequestHandlerBase`` registered
    under the matching ``RequestType``.

    Each generated method just calls :meth:`_dispatch` with the right
    ``RequestType``; keeping one implementation avoids 36 near-identical
    thunks in this file.  gRPC's method routing already runs before we
    get here, so ``_dispatch`` is the whole request path.
    """

    def __init__(
        self,
        handlers: dict[RequestType, RequestHandlerBase[Any]],
    ):
        self._handlers = handlers

    def _run_handler(
        self,
        request_type: RequestType,
        payloads: list[bytes],
        peer: str,
    ) -> Any:
        """Route a legacy-envelope payload list into the registered
        ``RequestHandlerBase`` and return the raw Python result.

        Split out of the msgspec path so the typed path can share the
        same executor / affinity dispatch without duplicating it.
        """
        handler = self._handlers.get(request_type)
        if handler is None:
            raise RuntimeError(f"No handler registered for {request_type}")

        handler_type = handler.get_handler_type()
        if handler_type is HandlerType.SYNC:
            assert isinstance(handler, SyncRequestHandler)
            return handler(payloads)
        if handler_type is HandlerType.BLOCKING:
            assert isinstance(handler, BlockingRequestHandler)
            # Peer id keeps the same affinity semantics as the old zmq
            # DEALER-ROUTER identity: one thread per client, forever.
            fut = handler(payloads, affinity_key=hash(peer))
            return fut.result()
        raise NotImplementedError(f"handler_type {handler_type} not supported")

    def _dispatch_typed(
        self,
        request: Any,
        context: "grpc.ServicerContext",
        request_type: RequestType,
        spec: TypedRpcSpec,
    ) -> Any:
        """Typed-rpc entry point.  Shares the executor / affinity logic
        with ``_dispatch`` via ``_run_handler``; the only difference is
        the wire format on either end.

        The registered ``RequestHandlerBase`` still speaks the msgspec
        payload-list ABI internally (business handlers haven't changed),
        so we re-encode the unpacked positional args back to msgspec
        bytes here.  That's a temporary crutch -- once every rpc is
        typed, ``RequestHandlerBase`` itself will lose the ``list[bytes]``
        parameter and take positional Python args directly.
        """
        handler = self._handlers.get(request_type)
        if handler is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"No handler registered for {request_type}",
            )
            raise RuntimeError("unreachable")

        py_args = spec.request_to_python(request)
        payload_classes = get_payload_classes(request_type)
        if len(py_args) != len(payload_classes):
            context.abort(
                grpc.StatusCode.INTERNAL,
                (
                    f"typed rpc {request_type} produced {len(py_args)} args, "
                    f"but protocol expects {len(payload_classes)}"
                ),
            )
            raise RuntimeError("unreachable")
        b_payloads = [
            msgspec_encode(arg, cls=cls)
            for arg, cls in zip(py_args, payload_classes, strict=False)
        ]
        result = self._run_handler(request_type, b_payloads, context.peer())
        return spec.python_to_response(result)


def _install_servicer_methods() -> None:
    """Attach one typed dispatch method per ``RequestType`` to the servicer."""
    for rt in RequestType:
        method_name = request_type_to_method_name(rt)
        typed_spec = _TYPED_RPCS[rt]
        method: Callable[..., Any]
        _resolved_spec: TypedRpcSpec = typed_spec

        def _typed_method(  # noqa: E501 (captured ``rt`` / ``spec`` via default arg)
            self: _RequestHandlerServicer,
            request: Any,
            context: "grpc.ServicerContext",
            _rt: RequestType = rt,
            _spec: TypedRpcSpec = _resolved_spec,
        ) -> Any:
            return self._dispatch_typed(request, context, _rt, _spec)

        method = _typed_method
        method.__name__ = method_name
        method.__qualname__ = f"_RequestHandlerServicer.{method_name}"
        setattr(_RequestHandlerServicer, method_name, method)


_install_servicer_methods()


# ---------------------------------------------------------------------------
# Server: public MessageQueueServer API preserved
# ---------------------------------------------------------------------------


@dataclass
class _ServerConfig:
    bind_url: str
    max_concurrency: int = 32


class MessageQueueServer:
    """gRPC server that wraps ``RequestHandlerBase`` instances.

    Public API mirrors the historical zmq-backed one so no module needs
    to change: ``add_handler`` / ``add_normal_thread_pool`` /
    ``add_affinity_thread_pool`` / ``start`` / ``close`` all keep their
    old semantics.

    Args:
        bind_url: Either ``grpc://host:port`` or a bare ``host:port``.
        context: Legacy positional slot (used to be zmq.Context); ignored.
        transport: Legacy positional slot; ignored.
        grpc_max_workers: Size of the base grpc thread pool.  Sync
            handlers run here directly; blocking handlers hand off to
            their dedicated thread pool so this executor stays free
            for dispatch and shouldn't need many threads.
    """

    def __init__(
        self,
        bind_url: str,
        context: Optional[Any] = None,
        transport: Optional[Any] = None,
        grpc_max_workers: int = 32,
    ):
        del context, transport  # legacy positional slots, no longer used
        self._bind_url = bind_url
        self._grpc_max_workers = grpc_max_workers
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self._server: grpc.Server | None = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Handler registration (identical semantics to the old zmq server)
    # ------------------------------------------------------------------

    def _inspect_handler_signature(
        self, request_type: RequestType, handler: Callable[..., Any]
    ) -> bool:
        """Verify a handler's parameter / return annotations match the
        registered ``ProtocolDefinition``.

        Returns:
            True if the signature matches or the annotations are omitted
            in a way that keeps us backwards compatible; False otherwise.
        """

        def same_type(a: Any, b: Any) -> bool:
            if a is None:
                a = type(None)
            if b is None:
                b = type(None)
            return a == b

        sig = inspect.signature(handler)
        hints = get_type_hints(handler)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        payload_clss = get_payload_classes(request_type)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d args, but got %d",
                request_type,
                len(payload_clss),
                len(params),
            )
            return False

        for i, (param, expected_cls) in enumerate(
            zip(params, payload_clss, strict=False)
        ):
            ann = hints.get(param.name, param.annotation)
            if not same_type(ann, expected_cls):
                logger.error(
                    "Handler for %s arg %d expects %s, got %s",
                    request_type,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(request_type)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return %s, got %s",
                request_type,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler: Callable[..., Any],
    ) -> None:
        if not self._inspect_handler_signature(request_type, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {request_type}"
            )

        if handler_type is HandlerType.SYNC:
            self.add_sync_handler(request_type, payload_clss, handler)
        elif handler_type is HandlerType.BLOCKING:
            self.add_blocking_handler(request_type, payload_clss, handler)
        elif handler_type is HandlerType.NON_BLOCKING:
            raise NotImplementedError("Non-blocking handler is not supported yet")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = SyncRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_blocking_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = BlockingRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        request_types: list[RequestType],
        method_name: str,
    ) -> None:
        for request_type in request_types:
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(
                    f"No handler registered for request type: {request_type}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {request_type} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler."
                )

    def add_normal_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        self._validate_blocking_handlers(request_types, "add_normal_thread_pool")
        if not request_types:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        self._validate_blocking_handlers(request_types, "add_affinity_thread_pool")
        if not request_types:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        for rt, handler in self.handlers.items():
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    "assigned. Call add_normal_thread_pool or "
                    "add_affinity_thread_pool before start()."
                )

        target = _parse_grpc_url(self._bind_url)
        server = grpc.server(
            ThreadPoolExecutor(
                max_workers=self._grpc_max_workers,
                thread_name_prefix="mq-grpc-server",
            ),
            options=_GRPC_UNLIMITED_MSG_OPTS,
        )
        servicer = _RequestHandlerServicer(self.handlers)
        lmcache_mq_pb2_grpc.add_MessageQueueServicer_to_server(servicer, server)
        server.add_insecure_port(target)
        server.start()
        self._server = server
        logger.info("MessageQueueServer listening on %s (gRPC)", self._bind_url)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._server is not None:
            self._server.stop(grace=None)
            self._server = None
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
