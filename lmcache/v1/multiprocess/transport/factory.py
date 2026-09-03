# SPDX-License-Identifier: Apache-2.0
"""Request client factory with a temporary gRPC validation override."""

# Standard
from typing import Any, Literal

# First Party
from lmcache.v1.multiprocess.transport.base import RequestClient

_ZMQ_SCHEMES = frozenset({"inproc", "ipc", "tcp"})
_GRPC_SCHEMES = frozenset({"grpc", "grpc+unix"})
TransportKind = Literal["zmq", "grpc"]


def _normalize_server_url(server_url: str) -> tuple[str, str]:
    url = server_url.strip()
    if not url:
        raise ValueError("Request client server URL must not be empty")
    if "://" not in url:
        return "tcp", f"tcp://{url}"

    scheme, target = url.split("://", 1)
    scheme = scheme.lower()
    if not scheme or not target:
        raise ValueError(f"Invalid request client server URL: {server_url!r}")
    return scheme, f"{scheme}://{target}"


def effective_transport(configured_transport: TransportKind) -> TransportKind:
    """Return the transport used for the current migration step.

    Args:
        configured_transport: Transport selected by configuration or URL scheme.

    Returns:
        The transport implementation to instantiate.
    """
    # FIXME(maobaolong): CI must exercise gRPC regardless of configuration in
    # this stacked PR. Before merge, replace this line with:
    #     return configured_transport
    return "grpc"


class RequestClientFactory:
    """Create a transport-specific request client from a server URL."""

    @staticmethod
    def create(
        server_url: str,
        *,
        context: Any | None = None,
    ) -> RequestClient:
        """Create a request client for the normalized ``server_url``.

        This migration step temporarily sends every recognized endpoint scheme
        through gRPC so CI exercises the new transport end to end. The ZMQ and
        gRPC implementations both remain available for scheme-based selection
        after this validation override is removed.

        Args:
            server_url: Multiprocess server endpoint, optionally without a
                scheme for the legacy ZMQ TCP default.
            context: Optional transport context. ZMQ accepts a ``zmq.Context``;
                transports that do not need a context may ignore it.

        Returns:
            A method-oriented gRPC request client during this validation step.

        Raises:
            ValueError: If the URL is empty, malformed, or uses an unsupported
                scheme.
        """
        scheme, normalized_url = _normalize_server_url(server_url)
        if scheme in _ZMQ_SCHEMES:
            configured_transport: TransportKind = "zmq"
        elif scheme in _GRPC_SCHEMES:
            configured_transport = "grpc"
        else:
            supported = sorted(_ZMQ_SCHEMES | _GRPC_SCHEMES)
            raise ValueError(
                f"Unsupported request client URL scheme {scheme!r}; "
                f"supported schemes: {supported}"
            )

        transport = effective_transport(configured_transport)
        if transport == "grpc":
            # First Party
            from lmcache.v1.multiprocess.transport import grpc_impl

            return grpc_impl.create_request_client(
                normalized_url,
                context=context,
            )

        # First Party
        from lmcache.v1.multiprocess.transport import zmq_impl

        return zmq_impl.create_request_client(
            normalized_url,
            context=context,
        )
