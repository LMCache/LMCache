# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the ZMQ wire protocol."""

# First Party
from lmcache.v1.transport.zmq_wire import (
    LEGACY_OPERATION_IDS,
    ZMQ_ERROR_OPERATION,
    ZmqErrorResponse,
    decode_error_response,
    decode_operation,
    encode_error_response,
    encode_operation,
)

__all__ = [
    "LEGACY_OPERATION_IDS",
    "ZMQ_ERROR_OPERATION",
    "ZmqErrorResponse",
    "decode_error_response",
    "decode_operation",
    "encode_error_response",
    "encode_operation",
]
