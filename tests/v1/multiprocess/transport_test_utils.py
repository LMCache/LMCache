# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for request-transport integration tests."""

# Standard
from typing import Any, Literal
from urllib.parse import urlsplit

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

RequestTransport = Literal["zmq", "grpc"]
REQUEST_TRANSPORTS: tuple[RequestTransport, ...] = ("zmq", "grpc")


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


def request_server_config(
    transport: RequestTransport,
    server_url: str,
    *,
    max_cpu_workers: int = 4,
    max_gpu_workers: int = 1,
) -> MPServerConfig:
    """Build a request-server config from a loopback test URL.

    Args:
        transport: Request transport to exercise.
        server_url: TCP or gRPC loopback URL returned by request_server_url.
        max_cpu_workers: Worker count for blocking CPU-bound handlers.
        max_gpu_workers: Worker count for blocking affinity handlers.

    Returns:
        Multiprocess server config for create_request_server.

    Raises:
        ValueError: If the URL does not include a host and TCP port.
    """
    parsed = urlsplit(server_url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Test request URL must include host and port: {server_url}")
    return MPServerConfig(
        transport=transport,
        host=parsed.hostname,
        port=parsed.port,
        max_cpu_workers=max_cpu_workers,
        max_gpu_workers=max_gpu_workers,
    )


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
    mp_config = request_server_config(transport, server_url)
    server = create_request_server([lookup], mp_config)
    server.start()
    return server
