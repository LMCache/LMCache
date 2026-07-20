# SPDX-License-Identifier: Apache-2.0
"""Transport abstraction for LMCache mp-mode message queue.

The business layer (``mq.py``) speaks in opaque byte frames and delegates
all wire I/O to a :class:`ClientTransport` (engine side) or
:class:`ServerTransport` (server side).  Concrete transports are strategy
plug-ins registered by URL scheme via :mod:`.registry`.

The interfaces are intentionally minimal — only the actions that
``MessageQueueClient`` / ``MessageQueueServer`` in ``mq.py`` really need:

* client: connect, send_frames, recv_frames, readable_handle, close
* server: bind, recv_request, send_response, readable_handle, close

Framing (msgpack encode/decode, RequestType dispatch, ``MessagingFuture``
bookkeeping) all stays in ``mq.py``; transports never look inside the
frames.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# A transport-owned handle that can be registered with ``zmq.Poller``
# (or plain ``selectors``).  For zmq transports this is the underlying
# ``zmq.Socket``; for grpc / other transports this is typically an
# integer fd from a self-pipe / ``eventfd``.
PollHandle = Any


@dataclass(frozen=True)
class ClientContext:
    """Opaque per-connection identity produced by a :class:`ServerTransport`.

    Used by :class:`~lmcache.v1.multiprocess.mq.MessageQueueServer` as the
    affinity key so that requests from the same client are routed to the
    same worker thread in ``AffinityThreadPool``.  Must be hashable.

    Attributes:
        key: Transport-specific identity bytes.  For ZMQ this is the
            DEALER identity frame; for gRPC this is derived from the
            bidi-stream peer id.
    """

    key: bytes


@runtime_checkable
class ClientTransport(Protocol):
    """Engine-side transport.  Sends request frames, receives responses."""

    def connect(self, url: str) -> None:
        """Open the underlying connection to ``url``."""

    def send_frames(self, frames: list[bytes]) -> None:
        """Send one logical message composed of ``frames``.

        The transport is responsible for keeping the frames as a single
        logical unit on the wire (``send_multipart`` on zmq,
        ``MqRequest`` with repeated bytes on grpc, etc.).
        """

    def recv_frames(self) -> list[bytes] | None:
        """Return one incoming message's frames, or ``None`` if none ready.

        The call MUST NOT block; the shared polling loop calls it only
        after :meth:`readable_handle` becomes readable.
        """

    def readable_handle(self) -> PollHandle:
        """Return a handle registrable with ``zmq.Poller`` / ``selectors``.

        Readability of this handle implies at least one incoming message
        is available via :meth:`recv_frames`.
        """

    def close(self) -> None:
        """Release the underlying connection."""


@runtime_checkable
class ServerTransport(Protocol):
    """Server-side transport.  Accepts requests, sends responses."""

    def bind(self, url: str) -> None:
        """Bind to ``url`` and start accepting client connections."""

    def recv_request(self) -> tuple[ClientContext, list[bytes]] | None:
        """Return ``(client_ctx, frames)`` or ``None`` if none ready.

        ``client_ctx`` is an opaque handle the caller must pass back to
        :meth:`send_response` when replying, and can be hashed for
        affinity routing.
        """

    def send_response(self, client_ctx: ClientContext, frames: list[bytes]) -> None:
        """Send a response back to the client identified by ``client_ctx``."""

    def readable_handle(self) -> PollHandle:
        """Return a handle registrable with ``zmq.Poller`` / ``selectors``."""

    def close(self) -> None:
        """Release the underlying listener."""
