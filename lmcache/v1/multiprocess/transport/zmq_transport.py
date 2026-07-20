# SPDX-License-Identifier: Apache-2.0
"""ZMQ implementation of :class:`ClientTransport` / :class:`ServerTransport`.

This is the existing production code path relocated behind the transport
abstraction, with byte-for-byte identical wire behaviour:

* client side uses ``zmq.DEALER`` and ``send_multipart`` /
  ``recv_multipart``;
* server side uses ``zmq.ROUTER`` and treats the first frame of an
  incoming multipart message as the client's opaque identity.

The identity frame is wrapped in :class:`ClientContext` so business
code no longer references the raw bytes directly.
"""

# Standard
from typing import Optional

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.transport.base import (
    ClientContext,
    ClientTransport,
    ServerTransport,
)
from lmcache.v1.multiprocess.transport.registry import (
    register_client,
    register_server,
)


class ZmqClientTransport(ClientTransport):
    """``zmq.DEALER`` backed client transport."""

    def __init__(self, context: Optional[zmq.Context] = None) -> None:
        self._ctx = context if context is not None else zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None

    def connect(self, url: str) -> None:
        assert self._socket is None, "ZmqClientTransport already connected"
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.connect(url)

    def send_frames(self, frames: list[bytes]) -> None:
        assert self._socket is not None, "transport not connected"
        self._socket.send_multipart(frames)

    def recv_frames(self) -> list[bytes] | None:
        assert self._socket is not None, "transport not connected"
        return self._socket.recv_multipart()

    def readable_handle(self) -> zmq.Socket:
        assert self._socket is not None, "transport not connected"
        return self._socket

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class ZmqServerTransport(ServerTransport):
    """``zmq.ROUTER`` backed server transport."""

    def __init__(self, context: Optional[zmq.Context] = None) -> None:
        self._ctx = context if context is not None else zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None

    def bind(self, url: str) -> None:
        assert self._socket is None, "ZmqServerTransport already bound"
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.bind(url)

    def recv_request(self) -> tuple[ClientContext, list[bytes]] | None:
        assert self._socket is not None, "transport not bound"
        msg = self._socket.recv_multipart()
        if not msg:
            return None
        identity, *frames = msg
        return ClientContext(key=identity), frames

    def send_response(self, client_ctx: ClientContext, frames: list[bytes]) -> None:
        assert self._socket is not None, "transport not bound"
        self._socket.send_multipart([client_ctx.key, *frames])

    def readable_handle(self) -> zmq.Socket:
        assert self._socket is not None, "transport not bound"
        return self._socket

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------
# Both ipc:// and tcp:// URLs map to ZMQ transports today; adding a new
# transport (e.g. gRPC) is a matter of dropping a sibling module and
# calling ``@register_client(scheme) / @register_server(scheme)``.


@register_client("ipc")
@register_client("tcp")
def _make_zmq_client(
    context: Optional[zmq.Context] = None,
) -> ZmqClientTransport:
    return ZmqClientTransport(context=context)


@register_server("ipc")
@register_server("tcp")
def _make_zmq_server(
    context: Optional[zmq.Context] = None,
) -> ZmqServerTransport:
    return ZmqServerTransport(context=context)
