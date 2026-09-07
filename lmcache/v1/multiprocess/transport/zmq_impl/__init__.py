# SPDX-License-Identifier: Apache-2.0
"""ZMQ transport implementation for multiprocess requests."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.zmq_impl.client import (
    ZmqMultiprocessClient,
)


def create_request_client(
    server_url: str,
    *,
    context: Any | None = None,
) -> RequestClient:
    """Create a method-oriented request client backed by ZMQ.

    Args:
        server_url: ZMQ endpoint URL.
        context: Optional existing ``zmq.Context`` shared by the caller.

    Returns:
        A ZMQ-backed request client.
    """
    if context is None:
        # Third Party
        import zmq

        context = zmq.Context.instance()
    return ZmqMultiprocessClient(MessageQueueClient(server_url, context))


__all__ = ["ZmqMultiprocessClient", "create_request_client"]
