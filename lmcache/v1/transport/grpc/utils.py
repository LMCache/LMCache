# SPDX-License-Identifier: Apache-2.0
"""Channel options and error conversion shared by gRPC clients."""

# Standard
from typing import cast

# Third Party
import grpc

# First Party
from lmcache.v1.client.transport.errors import (
    MPRequestError,
    MPServerUnavailableError,
)

CONNECT_TIMEOUT_MS = 5000
INITIAL_RECONNECT_BACKOFF_MS = 100
MAX_RECONNECT_BACKOFF_MS = 1000

# Same bounds as the ZMQ client (#4671); gRPC defaults are 20 s and 120 s.
# "min_reconnect_backoff_ms" is gRPC's per-attempt connect timeout.
GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.min_reconnect_backoff_ms", CONNECT_TIMEOUT_MS),
    ("grpc.initial_reconnect_backoff_ms", INITIAL_RECONNECT_BACKOFF_MS),
    ("grpc.max_reconnect_backoff_ms", MAX_RECONNECT_BACKOFF_MS),
    # Connect directly, as ZMQ does. grpcio otherwise follows the proxy variables
    # that engines often have set for downloads, even to reach localhost.
    ("grpc.enable_http_proxy", 0),
)
_SERVER_GONE = frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED})


def request_error(exc: grpc.RpcError) -> MPServerUnavailableError | MPRequestError:
    """Convert a failed gRPC call to the shared request error contract.

    Args:
        exc: Failed gRPC call exposing its status code and details.

    Returns:
        A server-unavailable error for UNAVAILABLE or CANCELLED, otherwise a
        request error containing the peer's status code and details.
    """
    call = cast(grpc.Call, exc)
    code = call.code()
    error_cls = MPServerUnavailableError if code in _SERVER_GONE else MPRequestError
    return error_cls(code.name, call.details() or "")
