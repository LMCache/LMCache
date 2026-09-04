# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for request-transport integration tests."""

# Standard
from typing import Any, Literal

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.server import add_handler_helper
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services.lookup import (
    LookupServiceImpl,
)

RequestTransport = Literal["zmq", "grpc"]
REQUEST_TRANSPORTS: tuple[RequestTransport, ...] = ("zmq", "grpc")
RequestServer = MessageQueueServer | GrpcMultiprocessServer

_LOOKUP_HANDLERS = {
    RequestType.LOOKUP: "lookup",
    RequestType.QUERY_PREFETCH_STATUS: "query_prefetch_status",
    RequestType.WAIT_PREFETCH_STATUS: "wait_prefetch_status",
    RequestType.QUERY_PREFETCH_LOOKUP_HITS: "query_prefetch_lookup_hits",
    RequestType.FREE_LOOKUP_LOCKS: "free_lookup_locks",
    RequestType.END_SESSION: "end_session",
}


def request_server_url(transport: RequestTransport, port: int) -> str:
    """Build a loopback request URL for a test transport.

    Args:
        transport: Request transport to exercise.
        port: Loopback TCP port used by the test server.

    Returns:
        A transport-specific request URL.
    """
    scheme = "tcp" if transport == "zmq" else "grpc"
    return f"{scheme}://127.0.0.1:{port}"


def start_lookup_request_server(
    transport: RequestTransport,
    server_url: str,
    lookup: Any,
) -> RequestServer:
    """Start a minimal lookup service over the selected request transport.

    Args:
        transport: Request transport to exercise.
        server_url: Endpoint on which the server should listen.
        lookup: Object implementing the lookup methods used by the test.

    Returns:
        The started request server. The caller must close it.
    """
    if transport == "grpc":
        server = GrpcMultiprocessServer(
            server_url,
            max_cpu_workers=4,
            max_gpu_workers=1,
        )
        server.add_service("LookupService", LookupServiceImpl(lookup))
        server.start()
        return server

    server = MessageQueueServer(server_url, zmq.Context.instance())
    blocking_types: list[RequestType] = []
    for request_type, method_name in _LOOKUP_HANDLERS.items():
        handler = getattr(lookup, method_name, None)
        if not callable(handler):
            continue
        add_handler_helper(server, request_type, handler)
        blocking_types.append(request_type)
    server.add_normal_thread_pool(blocking_types, max_workers=4)
    server.start()
    return server
