# SPDX-License-Identifier: Apache-2.0
"""Operation identifiers and error envelopes for the ZMQ wire protocol."""

# Standard
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

# Third Party
import msgspec

ZMQ_ERROR_OPERATION = "__lmcache_rpc_error__"
"""Transport-reserved operation identifying a failed ZMQ RPC."""

RequestUID: TypeAlias = int
"""Request identifier carried in the first ZMQ message frame."""

SpecialCodec: TypeAlias = tuple[Any, Any]
"""An encoder/decoder pair exposing ``encode`` and ``decode`` methods."""


class ZmqPayloadCodec:
    """Encode typed ZMQ payload frames with explicitly injected special codecs."""

    def __init__(self, special_codecs: Mapping[Any, SpecialCodec] | None = None):
        self._special_codecs = special_codecs or {}

    def encode(self, obj: Any, cls: Any) -> bytes:
        """Encode ``obj`` using its declared wire type."""
        if cls in self._special_codecs:
            encoder, _ = self._special_codecs[cls]
            return encoder.encode(obj)
        if cls in (bool, int):
            obj = cls(obj)
        return msgspec.msgpack.encode(obj)

    def decode(self, data: bytes, cls: Any) -> Any:
        """Decode one payload frame using its declared wire type."""
        if cls in self._special_codecs:
            _, decoder = self._special_codecs[cls]
            return decoder.decode(data)
        if cls in (bool, int):
            return cls(msgspec.msgpack.decode(data))
        return msgspec.msgpack.decode(data, type=cls)

    def decode_payloads(
        self, payloads: list[bytes], payload_types: list[Any]
    ) -> list[Any]:
        """Decode a request's payload frames after validating their count."""
        if len(payloads) != len(payload_types):
            raise ValueError("Payload count does not match expected count")
        return [
            self.decode(payload, cls)
            for payload, cls in zip(payloads, payload_types, strict=True)
        ]


DEFAULT_ZMQ_PAYLOAD_CODEC = ZmqPayloadCodec()


def encode_request_uid(uid: RequestUID) -> bytes:
    """Encode a request identifier as one MessagePack frame."""
    return DEFAULT_ZMQ_PAYLOAD_CODEC.encode(uid, RequestUID)


def decode_request_uid(data: bytes) -> RequestUID:
    """Decode a request identifier frame."""
    return DEFAULT_ZMQ_PAYLOAD_CODEC.decode(data, RequestUID)


class ZmqErrorResponse(msgspec.Struct, frozen=True):
    """Describe a request rejected by the remote ZMQ server.

    Attributes:
        code: Stable machine-readable error category.
        operation: RPC operation rejected by the server.
        message: Human-readable explanation of the rejection.
    """

    code: str
    operation: str
    message: str


def encode_error_response(response: ZmqErrorResponse) -> bytes:
    """Encode a ZMQ RPC error payload.

    Args:
        response: Structured error returned by the server.

    Returns:
        The MessagePack representation of ``response``.
    """
    return msgspec.msgpack.encode(response)


def decode_error_response(data: bytes) -> ZmqErrorResponse:
    """Decode a ZMQ RPC error payload.

    Args:
        data: MessagePack-encoded error payload.

    Returns:
        The decoded structured error.

    Raises:
        msgspec.DecodeError: If ``data`` is not a valid error payload.
    """
    return msgspec.msgpack.decode(data, type=ZmqErrorResponse)


# RequestType emitted these IDs before operation-name routing. Keep the table
# frozen; new operations use their string name and must not be added here.
# ID 11 (query_prefetch_lookup_hits) is retired and must not be reused.
LEGACY_OPERATION_IDS = MappingProxyType(
    {
        "register_kv_cache": 1,
        "unregister_kv_cache": 2,
        "register_q_cache": 3,
        "unregister_q_cache": 4,
        "store_q": 5,
        "store": 6,
        "retrieve": 7,
        "lookup": 8,
        "query_prefetch_status": 9,
        "wait_prefetch_status": 10,
        "free_lookup_locks": 12,
        "end_session": 13,
        "register_kv_cache_engine_driven_context": 14,
        "unregister_kv_cache_engine_driven_context": 15,
        "prepare_store": 16,
        "commit_store": 17,
        "prepare_retrieve": 18,
        "commit_retrieve": 19,
        "clear": 20,
        "get_chunk_size": 21,
        "ping": 22,
        "report_block_allocation": 23,
        "noop": 24,
        "cb_register_rope": 25,
        "cb_unregister_rope": 26,
        "cb_retrieve_pre_computed": 27,
        "cb_unified_lookup": 28,
        "p2p_lookup_and_lock": 29,
        "p2p_query_lookup_results": 30,
        "p2p_unlock_objects": 31,
        "get_experimental": 32,
        "cb_protocol_handshake": 33,
    }
)
_LEGACY_ID_TO_OPERATION = {value: key for key, value in LEGACY_OPERATION_IDS.items()}


def encode_operation(operation: str) -> bytes:
    """Encode an operation as its legacy ID or stable string name."""
    wire_value: int | str = LEGACY_OPERATION_IDS.get(operation, operation)
    return msgspec.msgpack.encode(wire_value)


def decode_operation(data: bytes) -> str:
    """Decode a legacy integer ID or string operation name."""
    wire_value: Any = msgspec.msgpack.decode(data)
    if isinstance(wire_value, int):
        try:
            return _LEGACY_ID_TO_OPERATION[wire_value]
        except KeyError as exc:
            raise ValueError(f"Unknown legacy ZMQ operation id: {wire_value}") from exc
    if isinstance(wire_value, str):
        return wire_value
    raise TypeError(f"Invalid ZMQ operation identifier: {wire_value!r}")
