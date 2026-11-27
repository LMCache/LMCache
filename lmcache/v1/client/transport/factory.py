# SPDX-License-Identifier: Apache-2.0
"""Multiprocess gRPC request client factory."""

# First Party
from lmcache.v1.client.transport import grpc_impl
from lmcache.v1.transport.request_client import RequestClient


def _normalize_server_url(server_url: str) -> str:
    """Normalize a multiprocess request endpoint to a gRPC URL.

    Args:
        server_url: A bare ``host:port`` or a URL with an explicit scheme.

    Returns:
        A ``grpc://`` or ``grpc+unix://`` URL. Bare addresses and the
        ``tcp://`` alias use ``grpc://``.

    Raises:
        ValueError: If the address is empty, or an explicit URL has an
            unsupported scheme or is missing its scheme or target.
    """
    url = server_url.strip()
    if not url:
        raise ValueError("Request client server URL must not be empty")
    if "://" not in url:
        return f"grpc://{url}"

    scheme, target = url.split("://", 1)
    scheme = scheme.lower()
    if not scheme or not target:
        raise ValueError(f"Invalid request client server URL: {server_url!r}")
    if scheme == "tcp":
        scheme = "grpc"
    if scheme in ("ipc", "inproc"):
        raise ValueError(
            f"The {scheme}:// ZMQ request transport was removed. "
            "Use grpc://host:port and start the server with --transport grpc."
        )
    if scheme not in ("grpc", "grpc+unix"):
        raise ValueError(
            f"Unsupported request client URL scheme {scheme!r}. "
            "Use grpc://host:port, tcp://host:port or grpc+unix:///path."
        )
    return f"{scheme}://{target}"


class RequestClientFactory:
    """Create a gRPC request client from a server URL."""

    @staticmethod
    def create(
        server_url: str,
        *,
        context: object | None = None,
        api_key: str | None = None,
    ) -> RequestClient:
        """Create a gRPC request client for a multiprocess server.

        Bare ``host:port`` and ``tcp://host:port`` use ``grpc://host:port``.
        Use ``grpc+unix:///path`` for a gRPC Unix socket.

        Args:
            server_url: Multiprocess server endpoint, optionally without a
                scheme for the gRPC default.
            context: Accepted for compatibility with callers that supply a ZMQ
                context. gRPC ignores it.
            api_key: Optional request credential. When omitted, the client
                reads the shared key environment variables.

        Returns:
            A method-oriented gRPC request client.

        Raises:
            ValueError: If the URL is empty, malformed, or uses an unsupported
                scheme.
        """
        return grpc_impl.create_request_client(
            _normalize_server_url(server_url), context=context, api_key=api_key
        )
