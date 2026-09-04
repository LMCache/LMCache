# SPDX-License-Identifier: Apache-2.0
"""gRPC transport implementation for multiprocess requests."""

# Standard
from typing import Any, NoReturn


def create_request_client(
    server_url: str,
    *,
    context: Any | None = None,
) -> NoReturn:
    """Report that the gRPC request client has not been introduced yet.

    Args:
        server_url: gRPC endpoint selected by the request client factory.
        context: Unused optional transport context.

    Raises:
        NotImplementedError: Always, until the gRPC implementation lands.
    """
    del context
    raise NotImplementedError(
        f"gRPC request client for {server_url!r} is not available yet"
    )


__all__ = ["create_request_client"]
