# SPDX-License-Identifier: Apache-2.0
"""gRPC transport implementation for multiprocess requests."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.transport.base import RequestClient


def create_request_client(
    server_url: str,
    *,
    context: Any | None = None,
) -> RequestClient:
    """Create a generated-stub gRPC request client.

    Args:
        server_url: gRPC endpoint selected by the request client factory.
        context: Ignored; accepted for parity with other transports.

    Returns:
        A method-oriented gRPC request client.
    """
    del context

    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl.client import (
        GrpcMultiprocessClient,
    )

    # Descriptor-derived methods are installed on the class at import time.
    return GrpcMultiprocessClient(server_url)  # type: ignore[abstract]


__all__ = ["create_request_client"]
