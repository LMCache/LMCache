# SPDX-License-Identifier: Apache-2.0
"""Conversions between the api types and the generated protobuf messages.

The servicer decodes requests and encodes replies; the client does the
reverse. Decoding a reply raises ``OrchestratorError`` when it carries a
status this client does not know.
"""

# Standard
from collections.abc import Mapping
from typing import TypeVar

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

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")

_WRITE_STATUS_TO_PROTO: dict[WriteStatus, pb2.WriteGrant.Status] = {
    WriteStatus.WRITE_GRANTED: pb2.WriteGrant.WRITE_GRANTED,
    WriteStatus.EXISTS_VALID: pb2.WriteGrant.EXISTS_VALID,
    WriteStatus.BUSY_WRITING: pb2.WriteGrant.BUSY_WRITING,
    WriteStatus.OUT_OF_SPACE: pb2.WriteGrant.OUT_OF_SPACE,
}
_READ_STATUS_TO_PROTO: dict[ReadStatus, pb2.ReadGrant.Status] = {
    ReadStatus.READ_GRANTED: pb2.ReadGrant.READ_GRANTED,
    ReadStatus.MISS: pb2.ReadGrant.MISS,
    ReadStatus.BUSY_WRITING: pb2.ReadGrant.BUSY_WRITING,
}
_TOKEN_STATUS_TO_PROTO: dict[TokenStatus, pb2.ResultBatch.Status] = {
    TokenStatus.OK: pb2.ResultBatch.OK,
    TokenStatus.STALE_TOKEN: pb2.ResultBatch.STALE_TOKEN,
}
_WRITE_STATUS_FROM_PROTO = {wire: api for api, wire in _WRITE_STATUS_TO_PROTO.items()}
_READ_STATUS_FROM_PROTO = {wire: api for api, wire in _READ_STATUS_TO_PROTO.items()}
_TOKEN_STATUS_FROM_PROTO = {wire: api for api, wire in _TOKEN_STATUS_TO_PROTO.items()}


def _decode_status(
    table: Mapping[_KeyT, _ValueT], value: _KeyT, message_name: str
) -> _ValueT:
    """Map a wire status to its api enum.

    Raises:
        OrchestratorError: If the status is unspecified or unknown.
    """
    try:
        return table[value]
    except KeyError:
        raise OrchestratorError(
            f"orchestrator replied with unknown {message_name} status {value}"
        ) from None


def envelope_from_proto(msg: pb2.Envelope) -> Envelope:
    """Convert a wire envelope."""
    return Envelope(
        region_id=msg.region_id,
        expected_region_epoch=msg.expected_region_epoch,
        client_id=msg.client_id,
        client_incarnation=msg.client_incarnation,
        request_id=msg.request_id,
    )


def key_to_proto(key: WireObjectKey) -> pb2.ObjectKey:
    """Convert an object key to its wire form."""
    return pb2.ObjectKey(
        chunk_hash=key.chunk_hash,
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=key.cache_salt,
    )


def key_from_proto(msg: pb2.ObjectKey) -> WireObjectKey:
    """Convert a wire object key."""
    return WireObjectKey(
        chunk_hash=msg.chunk_hash,
        model_name=msg.model_name,
        kv_rank=msg.kv_rank,
        object_group_id=msg.object_group_id,
        cache_salt=msg.cache_salt,
    )


def layout_to_proto(layout: WireLayout) -> pb2.ObjectLayout:
    """Convert an object layout to its wire form."""
    return pb2.ObjectLayout(
        shapes=[pb2.TensorShape(dims=shape) for shape in layout.shapes],
        dtypes=layout.dtypes,
    )


def layout_from_proto(msg: pb2.ObjectLayout) -> WireLayout:
    """Convert a wire object layout."""
    return WireLayout(
        shapes=tuple(tuple(shape.dims) for shape in msg.shapes),
        dtypes=tuple(msg.dtypes),
    )


def handle_to_proto(handle: Handle) -> pb2.Handle:
    """Convert an extent handle to its wire form."""
    return pb2.Handle(
        offset=handle.offset, length=handle.length, generation=handle.generation
    )


def handle_from_proto(msg: pb2.Handle) -> Handle:
    """Convert a wire extent handle."""
    return Handle(offset=msg.offset, length=msg.length, generation=msg.generation)


def write_request_to_proto(entry: WriteRequest) -> pb2.WriteEntry:
    """Convert a write request to its wire entry."""
    return pb2.WriteEntry(
        key=key_to_proto(entry.key),
        payload_bytes=entry.payload_bytes,
        layout=layout_to_proto(entry.layout),
    )


def write_request_from_proto(msg: pb2.WriteEntry) -> WriteRequest:
    """Convert a wire write entry."""
    return WriteRequest(
        key=key_from_proto(msg.key),
        payload_bytes=msg.payload_bytes,
        layout=layout_from_proto(msg.layout),
    )


def read_request_to_proto(entry: ReadRequest) -> pb2.ReadEntry:
    """Convert a read request to its wire entry."""
    return pb2.ReadEntry(key=key_to_proto(entry.key), reader_count=entry.reader_count)


def read_request_from_proto(msg: pb2.ReadEntry) -> ReadRequest:
    """Convert a wire read entry."""
    return ReadRequest(key=key_from_proto(msg.key), reader_count=msg.reader_count)


def write_grant_to_proto(grant: WriteGrantResult) -> pb2.WriteGrant:
    """Convert a write grant to its wire form; handle and token only when
    granted."""
    return pb2.WriteGrant(
        status=_WRITE_STATUS_TO_PROTO[grant.status],
        handle=None if grant.handle is None else handle_to_proto(grant.handle),
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
        status, handle=handle_from_proto(msg.handle), token=msg.token
    )


def read_grant_to_proto(grant: ReadGrantResult) -> pb2.ReadGrant:
    """Convert a read grant to its wire form; the other fields only when
    granted."""
    return pb2.ReadGrant(
        status=_READ_STATUS_TO_PROTO[grant.status],
        handle=None if grant.handle is None else handle_to_proto(grant.handle),
        leases=grant.leases,
        layout=None if grant.layout is None else layout_to_proto(grant.layout),
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
        handle=handle_from_proto(msg.handle),
        leases=tuple(msg.leases),
        layout=layout_from_proto(msg.layout),
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
    return pb2.RegionContract(
        region_id=contract.region_id,
        region_epoch=contract.region_epoch,
        capacity_bytes=contract.capacity_bytes,
        alignment_bytes=contract.alignment_bytes,
        layout_fingerprint=contract.layout_fingerprint,
        visibility_mode=contract.visibility_mode,
        max_batch_entries=contract.max_batch_entries,
        reset_required=contract.reset_required,
    )


def contract_from_proto(msg: pb2.RegionContract) -> RegionContract:
    """Convert a wire region contract."""
    return RegionContract(
        region_id=msg.region_id,
        region_epoch=msg.region_epoch,
        capacity_bytes=msg.capacity_bytes,
        alignment_bytes=msg.alignment_bytes,
        layout_fingerprint=msg.layout_fingerprint,
        visibility_mode=msg.visibility_mode,
        max_batch_entries=msg.max_batch_entries,
        reset_required=msg.reset_required,
    )


def usage_to_proto(usage: RegionUsage) -> pb2.RegionUsage:
    """Convert a usage snapshot to its wire form."""
    return pb2.RegionUsage(
        capacity_bytes=usage.capacity_bytes,
        allocated_bytes=usage.allocated_bytes,
        valid_bytes=usage.valid_bytes,
        writing=usage.writing,
        valid=usage.valid,
        consumed=usage.consumed,
        read_leases=usage.read_leases,
        clients=usage.clients,
    )


def usage_from_proto(msg: pb2.RegionUsage) -> RegionUsage:
    """Convert a wire usage snapshot."""
    return RegionUsage(
        capacity_bytes=msg.capacity_bytes,
        allocated_bytes=msg.allocated_bytes,
        valid_bytes=msg.valid_bytes,
        writing=msg.writing,
        valid=msg.valid,
        consumed=msg.consumed,
        read_leases=msg.read_leases,
        clients=msg.clients,
    )
