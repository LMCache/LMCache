# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for request-transport integration tests."""

# Standard
from typing import Any, Literal

# Third Party
import zmq

from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.zmq_impl.server import (
    ThreadPoolType,
    add_handler_helper,
    get_zmq_handler_specs,
)

RequestTransport = Literal["zmq", "grpc"]
REQUEST_TRANSPORTS: tuple[RequestTransport, ...] = ("zmq", "grpc")
RequestServer = MessageQueueServer | GrpcMultiprocessServer


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
        grpc_server = GrpcMultiprocessServer(
            server_url,
            max_cpu_workers=4,
            max_gpu_workers=1,
        )
        grpc_server.add_modules([lookup])
        grpc_server.start()
        return grpc_server

    zmq_server = MessageQueueServer(server_url, zmq.Context.instance())
    specs = get_zmq_handler_specs(lookup)
    for spec in specs:
        add_handler_helper(
            zmq_server,
            spec.operation,
            spec.handler,
            spec.handler_type,
        )
    normal_operations = [
        spec.operation for spec in specs if spec.pool is ThreadPoolType.NORMAL
    ]
    if normal_operations:
        zmq_server.add_normal_thread_pool(normal_operations, max_workers=4)
    zmq_server.start()
    return zmq_server
