# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for request-transport integration tests."""

# Standard
from typing import Any, Literal, Protocol
import importlib

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.transport.zmq_impl.server import add_handler_helper

RequestTransport = Literal["zmq", "grpc"]

# Keep the gRPC type and URL path ready, but do not execute gRPC tests until
# the runtime implementation lands. Enabling both transports is a one-line
# change to this tuple.
REQUEST_TRANSPORTS: tuple[RequestTransport, ...] = ("zmq",)


class RequestServer(Protocol):
    """Common lifecycle used by request-transport test servers."""

    def close(self) -> None: ...


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
        grpc_server_module = importlib.import_module(
            "lmcache.v1.multiprocess.transport.grpc_impl.server"
        )
        grpc_server_class = vars(grpc_server_module)["GrpcMultiprocessServer"]
        grpc_server = grpc_server_class(
            server_url,
            max_cpu_workers=4,
            max_gpu_workers=1,
        )
        grpc_server.add_modules([lookup])
        grpc_server.start()
        return grpc_server

    zmq_server = MessageQueueServer(server_url, zmq.Context.instance())
    blocking_types: list[RequestType] = []
    for request_type, method_name in _LOOKUP_HANDLERS.items():
        handler = getattr(lookup, method_name, None)
        if not callable(handler):
            continue
        add_handler_helper(zmq_server, request_type, handler)
        blocking_types.append(request_type)
    if blocking_types:
        zmq_server.add_normal_thread_pool(blocking_types, max_workers=4)
    zmq_server.start()
    return zmq_server
