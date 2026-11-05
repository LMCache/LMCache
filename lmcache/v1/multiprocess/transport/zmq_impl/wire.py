# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the ZMQ wire protocol."""

# First Party
from lmcache.v1.transport.zmq_wire import (
    DEFAULT_ZMQ_PAYLOAD_CODEC,
    LEGACY_OPERATION_IDS,
    ZMQ_ERROR_OPERATION,
    RequestUID,
    ZmqErrorResponse,
    ZmqPayloadCodec,
    decode_error_response,
    decode_operation,
    decode_request_uid,
    encode_error_response,
    encode_operation,
    encode_request_uid,
)

__all__ = [
    "LEGACY_OPERATION_IDS",
    "DEFAULT_ZMQ_PAYLOAD_CODEC",
    "RequestUID",
    "ZMQ_ERROR_OPERATION",
    "ZmqErrorResponse",
    "ZmqPayloadCodec",
    "decode_error_response",
    "decode_operation",
    "decode_request_uid",
    "encode_error_response",
    "encode_operation",
    "encode_request_uid",
]
