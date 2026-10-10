# SPDX-License-Identifier: Apache-2.0
"""Conversions between the api types and the generated protobuf messages."""

# Standard
from collections.abc import Mapping
from dataclasses import asdict
from typing import TypeVar

# Third Party
from google.protobuf.message import Message

# First Party
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator.api import (
    Envelope,
    Handle,
    OrchestratorError,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionUsage,
    TokenStatus,
    WireLayout,
    WireObjectKey,
    WriteGrantResult,
    WriteRequest,
    WriteStatus,
)

_WireT = TypeVar("_WireT")
_ApiT = TypeVar("_ApiT")

# Every api status maps to the proto enum value of the same name.
_WRITE_STATUS_TO_PROTO = {s: pb2.WriteGrant.Status.Value(s.name) for s in WriteStatus}
_READ_STATUS_TO_PROTO = {s: pb2.ReadGrant.Status.Value(s.name) for s in ReadStatus}
_TOKEN_STATUS_TO_PROTO = {s: pb2.ResultBatch.Status.Value(s.name) for s in TokenStatus}
_WRITE_STATUS_FROM_PROTO = {wire: api for api, wire in _WRITE_STATUS_TO_PROTO.items()}
_READ_STATUS_FROM_PROTO = {wire: api for api, wire in _READ_STATUS_TO_PROTO.items()}
_TOKEN_STATUS_FROM_PROTO = {wire: api for api, wire in _TOKEN_STATUS_TO_PROTO.items()}


def _from_proto(cls: type[_ApiT], msg: Message) -> _ApiT:
    """Build the api dataclass ``cls`` from the same-named fields of ``msg``."""
    return cls(**{f.name: getattr(msg, f.name) for f in msg.DESCRIPTOR.fields})


# Keys and handles are converted once per batch entry, so they spell out their
# fields; the generic helper costs about twice as much per entry.
def _key_to_proto(key: WireObjectKey) -> pb2.ObjectKey:
    """Convert an object key to its wire form."""
    return pb2.ObjectKey(
        chunk_hash=key.chunk_hash,
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=key.cache_salt,
    )


def _key_from_proto(msg: pb2.ObjectKey) -> WireObjectKey:
    """Convert a wire object key."""
    return WireObjectKey(
        chunk_hash=msg.chunk_hash,
        model_name=msg.model_name,
        kv_rank=msg.kv_rank,
        object_group_id=msg.object_group_id,
        cache_salt=msg.cache_salt,
    )


def _handle_to_proto(handle: Handle | None) -> pb2.Handle | None:
    """Convert an extent handle to its wire form."""
    if handle is None:
        return None
    return pb2.Handle(
        offset=handle.offset, length=handle.length, generation=handle.generation
    )


def _handle_from_proto(msg: pb2.Handle) -> Handle:
    """Convert a wire extent handle."""
    return Handle(offset=msg.offset, length=msg.length, generation=msg.generation)


def _layout_to_proto(layout: WireLayout) -> pb2.ObjectLayout:
    """Convert an object layout to its wire form."""
    return pb2.ObjectLayout(
        shapes=[pb2.TensorShape(dims=shape) for shape in layout.shapes],
        dtypes=layout.dtypes,
    )


def _layout_from_proto(msg: pb2.ObjectLayout) -> WireLayout:
    """Convert a wire object layout."""
    return WireLayout(
        shapes=tuple(tuple(shape.dims) for shape in msg.shapes),
        dtypes=tuple(msg.dtypes),
    )


def _decode_status(
    table: Mapping[_WireT, _ApiT], value: _WireT, message_name: str
) -> _ApiT:
    """Map a wire status to its api enum; raise ``OrchestratorError`` if unknown."""
    try:
        return table[value]
    except KeyError:
        raise OrchestratorError(
            f"orchestrator replied with unknown {message_name} status {value}"
        ) from None


def envelope_from_proto(msg: pb2.Envelope) -> Envelope:
    """Convert a wire envelope."""
    return _from_proto(Envelope, msg)


def write_request_to_proto(entry: WriteRequest) -> pb2.WriteEntry:
    """Convert a write request to its wire entry."""
    return pb2.WriteEntry(
        key=_key_to_proto(entry.key),
        payload_bytes=entry.payload_bytes,
        layout=_layout_to_proto(entry.layout),
    )


def write_request_from_proto(msg: pb2.WriteEntry) -> WriteRequest:
    """Convert a wire write entry."""
    return WriteRequest(
        key=_key_from_proto(msg.key),
        payload_bytes=msg.payload_bytes,
        layout=_layout_from_proto(msg.layout),
    )


def read_request_to_proto(entry: ReadRequest) -> pb2.ReadEntry:
    """Convert a read request to its wire entry."""
    return pb2.ReadEntry(key=_key_to_proto(entry.key), reader_count=entry.reader_count)


def read_request_from_proto(msg: pb2.ReadEntry) -> ReadRequest:
    """Convert a wire read entry."""
    return ReadRequest(key=_key_from_proto(msg.key), reader_count=msg.reader_count)


def write_grant_to_proto(grant: WriteGrantResult) -> pb2.WriteGrant:
    """Convert a write grant to its wire form; handle and token only when granted."""
    return pb2.WriteGrant(
        status=_WRITE_STATUS_TO_PROTO[grant.status],
        handle=_handle_to_proto(grant.handle),
        token=grant.token,
    )


def write_grant_from_proto(msg: pb2.WriteGrant) -> WriteGrantResult:
    """Convert a wire write grant.

    Raises:
        OrchestratorError: If the status is unknown.
    """
    status = _decode_status(_WRITE_STATUS_FROM_PROTO, msg.status, "WriteGrant")
    if status is not WriteStatus.WRITE_GRANTED:
        return WriteGrantResult(status)
    return WriteGrantResult(
        status, handle=_handle_from_proto(msg.handle), token=msg.token
    )


def read_grant_to_proto(grant: ReadGrantResult) -> pb2.ReadGrant:
    """Convert a read grant to its wire form; the other fields only when granted."""
    return pb2.ReadGrant(
        status=_READ_STATUS_TO_PROTO[grant.status],
        handle=_handle_to_proto(grant.handle),
        leases=grant.leases,
        layout=None if grant.layout is None else _layout_to_proto(grant.layout),
        payload_bytes=grant.payload_bytes,
    )


def read_grant_from_proto(msg: pb2.ReadGrant) -> ReadGrantResult:
    """Convert a wire read grant.

    Raises:
        OrchestratorError: If the status is unknown.
    """
    status = _decode_status(_READ_STATUS_FROM_PROTO, msg.status, "ReadGrant")
    if status is not ReadStatus.READ_GRANTED:
        return ReadGrantResult(status)
    return ReadGrantResult(
        status,
        handle=_handle_from_proto(msg.handle),
        leases=tuple(msg.leases),
        layout=_layout_from_proto(msg.layout),
        payload_bytes=msg.payload_bytes,
    )


def token_status_to_proto(status: TokenStatus) -> pb2.ResultBatch.Status:
    """Convert a token result to its wire form."""
    return _TOKEN_STATUS_TO_PROTO[status]


def token_status_from_proto(value: pb2.ResultBatch.Status) -> TokenStatus:
    """Convert a wire token result.

    Raises:
        OrchestratorError: If the status is unknown.
    """
    return _decode_status(_TOKEN_STATUS_FROM_PROTO, value, "ResultBatch")


def contract_to_proto(contract: RegionContract) -> pb2.RegionContract:
    """Convert a region contract to its wire form."""
    return pb2.RegionContract(**asdict(contract))


def contract_from_proto(msg: pb2.RegionContract) -> RegionContract:
    """Convert a wire region contract."""
    return _from_proto(RegionContract, msg)


def usage_to_proto(usage: RegionUsage) -> pb2.RegionUsage:
    """Convert a usage snapshot to its wire form."""
    return pb2.RegionUsage(**asdict(usage))


def usage_from_proto(msg: pb2.RegionUsage) -> RegionUsage:
    """Convert a wire usage snapshot."""
    return _from_proto(RegionUsage, msg)
