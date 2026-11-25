# SPDX-License-Identifier: Apache-2.0
"""Request client factory selected by endpoint URL scheme."""

# Standard
from typing import Any
import warnings

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.transport.request_client import RequestClient
from lmcache.v1.transport.zmq import ZMQ_MIGRATION_DOC_URL

logger = init_logger(__name__)

_ZMQ_SCHEMES = frozenset({"inproc", "ipc", "tcp"})
_GRPC_SCHEMES = frozenset({"grpc", "grpc+unix"})

_zmq_deprecation_warned = False


def _warn_zmq_deprecated(server_url: str) -> None:
    global _zmq_deprecation_warned
    if _zmq_deprecation_warned:
        return
    _zmq_deprecation_warned = True
    message = (
        f"The zmq request transport ({server_url}) is deprecated and will be "
        "removed in a future release. Use grpc://host:port and start the "
        f"server with --transport grpc. See {ZMQ_MIGRATION_DOC_URL}"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)


def _normalize_server_url(server_url: str) -> tuple[str, str]:
    url = server_url.strip()
    if not url:
        raise ValueError("Request client server URL must not be empty")
    if "://" not in url:
        return "grpc", f"grpc://{url}"

    scheme, target = url.split("://", 1)
    scheme = scheme.lower()
    if not scheme or not target:
        raise ValueError(f"Invalid request client server URL: {server_url!r}")
    return scheme, f"{scheme}://{target}"


class RequestClientFactory:
    """Create a transport-specific request client from a server URL."""

    @staticmethod
    def create(
        server_url: str,
        *,
        context: Any | None = None,
        api_key: str | None = None,
    ) -> RequestClient:
        """Create a request client selected by ``server_url`` scheme.

        Bare ``host:port`` endpoints default to ``grpc://``. gRPC handles
        ``grpc://`` and ``grpc+unix://``; the deprecated ZMQ transport handles
        ``tcp://``, ``ipc://``, and ``inproc://``.

        Args:
            server_url: Multiprocess server endpoint, optionally without a
                scheme for the gRPC default.
            context: Optional transport context. ZMQ accepts a ``zmq.Context``;
                transports that do not need a context may ignore it.
            api_key: Optional request-transport API key. If omitted, both
                transports fall back to the shared key environment variables.
                ZMQ carries the key in a wire authentication frame; gRPC
                carries it as request metadata.

        Returns:
            A method-oriented request client for the selected transport.

        Raises:
            ValueError: If the URL is empty, malformed, or uses an unsupported
                scheme.
        """
        scheme, normalized_url = _normalize_server_url(server_url)
        if scheme in _ZMQ_SCHEMES:
            _warn_zmq_deprecated(normalized_url)
            # First Party
            from lmcache.v1.client.transport import zmq_impl

            return zmq_impl.create_request_client(
                normalized_url,
                context=context,
                api_key=api_key,
            )
        if scheme in _GRPC_SCHEMES:
            # First Party
            from lmcache.v1.client.transport import grpc_impl

            return grpc_impl.create_request_client(
                normalized_url,
                context=context,
                api_key=api_key,
            )

        supported = sorted(_ZMQ_SCHEMES | _GRPC_SCHEMES)
        raise ValueError(
            f"Unsupported request client URL scheme {scheme!r}; "
            f"supported schemes: {supported}"
        )
