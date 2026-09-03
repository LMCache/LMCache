# SPDX-License-Identifier: Apache-2.0
"""ZMQ transport implementation for multiprocess requests."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.mq import DEFAULT_CONNECT_TIMEOUT, MessageQueueClient
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.zmq_impl.client import (
    ZmqMultiprocessClient,
)


def create_request_client(
    server_url: str,
    *,
    context: Any | None = None,
    connect_timeout: float | None = None,
) -> RequestClient:
    """Create a method-oriented request client backed by ZMQ.

    Args:
        server_url: ZMQ endpoint URL.
        context: Optional existing ``zmq.Context`` shared by the caller.
        connect_timeout: Optional bound (seconds) on each TCP connect attempt
            of the client socket; ``None`` keeps ``DEFAULT_CONNECT_TIMEOUT``.

    Returns:
        A ZMQ-backed request client.
    """
    if context is None:
        # Third Party
        import zmq

        context = zmq.Context.instance()
    if connect_timeout is None:
        connect_timeout = DEFAULT_CONNECT_TIMEOUT
    return ZmqMultiprocessClient(
        MessageQueueClient(server_url, context, connect_timeout=connect_timeout)
    )


__all__ = ["ZmqMultiprocessClient", "create_request_client"]
