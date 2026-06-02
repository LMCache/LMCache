# SPDX-License-Identifier: Apache-2.0
"""ZMQ implementation of the mp coordinator transport boundary.

Coordinator side binds two sockets -- a ROUTER for request/reply and a PULL for
fire-and-forget -- and keeps one REQ "command" socket per registered instance
for server-initiated push. Client side is synchronous: it opens a fresh REQ per
request (so a timeout never wedges a long-lived socket) and binds a REP socket
to serve pushed commands.
"""

# Standard
import asyncio

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.transport import (
    ClientCommandHandler,
    ClientTransport,
    CoordinatorTransport,
    Inbound,
    InboundHandler,
    InboundKind,
    ReachInfo,
    StopPredicate,
    TransportError,
    UnknownInstanceError,
)
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    close_zmq_socket,
    get_zmq_socket,
    get_zmq_socket_with_timeout,
)

logger = init_logger(__name__)

# Poll cadence for the client control socket so the serve loop can observe a
# stop request between blocking receives.
_CONTROL_POLL_TIMEOUT_MS = 500


def _open_command_socket(
    zmq_context: zmq.asyncio.Context, reach: ReachInfo, recv_timeout_ms: int
) -> zmq.asyncio.Socket:
    """Open a REQ command socket connected to an instance's control REP socket.

    A plain REQ socket (no ``REQ_CORRELATE``) is used so the wire framing stays
    compatible with the coordinator's ROUTER parsing.

    Args:
        zmq_context: The shared async ZMQ context.
        reach: Where to reach the instance.
        recv_timeout_ms: Receive timeout for command replies.

    Returns:
        A connected REQ socket with send/recv timeouts set.
    """
    return get_zmq_socket_with_timeout(
        zmq_context,
        f"{reach.ip}:{reach.control_port}",
        protocol="tcp",
        role=zmq.REQ,  # type: ignore[attr-defined]
        bind_or_connect="connect",
        recv_timeout_ms=recv_timeout_ms,
        send_timeout_ms=DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    )


class ZmqCoordinatorTransport(CoordinatorTransport):
    """ZMQ coordinator-side transport.

    Args:
        zmq_context: The shared async ZMQ context.
        request_url: Bind address for the request/reply ROUTER socket.
        push_url: Bind address for the fire-and-forget PULL socket.
        command_timeout_ms: Receive timeout for per-instance command replies.
    """

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        request_url: str,
        push_url: str,
        command_timeout_ms: int = DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    ) -> None:
        """Bind the request and push sockets."""
        self._zmq_context = zmq_context
        self._command_timeout_ms = command_timeout_ms
        self._request_socket = get_zmq_socket(
            zmq_context,
            request_url,
            protocol="tcp",
            role=zmq.ROUTER,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        self._push_socket = get_zmq_socket(
            zmq_context,
            push_url,
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        # Per-instance command sockets and their reach (for repair), plus a lock
        # per instance: a REQ socket cannot serve overlapping request/reply pairs.
        self._sockets: dict[str, zmq.asyncio.Socket] = {}
        self._reach: dict[str, ReachInfo] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def serve(self, handler: InboundHandler) -> None:
        """Serve the request and push sockets concurrently until cancelled.

        Args:
            handler: The coordinator core's dispatch callback.
        """
        await asyncio.gather(
            self._serve_requests(handler),
            self._serve_push(handler),
        )

    async def _serve_requests(self, handler: InboundHandler) -> None:
        """Receive request frames, dispatch, and reply on the ROUTER socket."""
        while True:
            frames = await self._request_socket.recv_multipart()
            if len(frames) < 3:
                logger.error(
                    "Invalid ROUTER frame count %d (expected >= 3)", len(frames)
                )
                continue
            identity = frames[0]
            payload = frames[2]
            try:
                reply = await handler(Inbound(InboundKind.REQUEST, payload))
            except Exception as e:
                logger.error("Request handler raised: %s", e)
                continue
            if reply is None:
                logger.error("Request handler returned no reply; dropping")
                continue
            await self._request_socket.send_multipart([identity, b"", reply])

    async def _serve_push(self, handler: InboundHandler) -> None:
        """Receive fire-and-forget messages on the PULL socket and dispatch."""
        while True:
            parts = await self._push_socket.recv_multipart()
            for part in parts:
                try:
                    await handler(Inbound(InboundKind.PUSH, part))
                except Exception as e:
                    logger.error("Push handler raised: %s", e)

    def _lock_for(self, instance_id: str) -> asyncio.Lock:
        """Return (creating if needed) the send lock for an instance."""
        lock = self._locks.get(instance_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[instance_id] = lock
        return lock

    async def send_command(self, instance_id: str, payload: bytes) -> bytes:
        """Push a command to one instance and return its reply.

        The per-instance lock serializes exchanges on the REQ socket. On failure
        the wedged socket is rebuilt in place before the error propagates, so a
        later command recovers.

        Args:
            instance_id: Target mp server.
            payload: Opaque, encoded command bytes.

        Returns:
            The instance's raw reply bytes.

        Raises:
            UnknownInstanceError: If the instance is not reachable.
            TransportError: If the exchange fails.
        """
        if instance_id not in self._sockets:
            raise UnknownInstanceError(f"Instance {instance_id} is not reachable")
        async with self._lock_for(instance_id):
            # Re-check under the lock: a deregister/eviction may have removed the
            # instance while we waited for it, and the contract is to raise
            # UnknownInstanceError (which broadcast skips), never a raw KeyError.
            socket = self._sockets.get(instance_id)
            if socket is None:
                raise UnknownInstanceError(f"Instance {instance_id} is not reachable")
            try:
                await socket.send(payload)
                return await socket.recv()
            except zmq.ZMQError as e:
                self._repair(instance_id)
                raise TransportError(f"Command to {instance_id} failed: {e}") from e

    def _repair(self, instance_id: str) -> None:
        """Replace a wedged command socket using the stored reach.

        A no-op if the instance was removed concurrently (no reach/socket left).
        """
        reach = self._reach.get(instance_id)
        socket = self._sockets.get(instance_id)
        if reach is None or socket is None:
            return
        close_zmq_socket(socket)
        self._sockets[instance_id] = _open_command_socket(
            self._zmq_context, reach, self._command_timeout_ms
        )
        logger.info("Reopened command socket for instance %s", instance_id)

    async def broadcast(self, payload: bytes) -> dict[str, bytes]:
        """Push a command to every reachable instance, skipping failures.

        Args:
            payload: Opaque, encoded command bytes.

        Returns:
            A mapping of instance id to reply for instances that replied.
        """
        instance_ids = list(self._sockets)
        if not instance_ids:
            return {}

        async def _send_one(instance_id: str) -> tuple[str, bytes] | None:
            try:
                return instance_id, await self.send_command(instance_id, payload)
            except (UnknownInstanceError, TransportError) as e:
                logger.warning("Broadcast to instance %s failed: %s", instance_id, e)
                return None

        results = await asyncio.gather(*(_send_one(i) for i in instance_ids))
        return {item[0]: item[1] for item in results if item is not None}

    def add_instance(self, instance_id: str, reach: ReachInfo) -> None:
        """Open (or replace) the command socket for an instance.

        Args:
            instance_id: The registering mp server's id.
            reach: How to reach it for push.

        Raises:
            TransportError: If the command socket could not be opened. On
                failure any existing reach is left intact (not torn down).
        """
        # Open the replacement first; only swap in (and close the old) on
        # success, so a failed re-register leaves the existing reach usable.
        try:
            socket = _open_command_socket(
                self._zmq_context, reach, self._command_timeout_ms
            )
        except zmq.ZMQError as e:
            raise TransportError(
                f"Cannot reach {reach.ip}:{reach.control_port}: {e}"
            ) from e
        old_socket = self._sockets.get(instance_id)
        self._sockets[instance_id] = socket
        self._reach[instance_id] = reach
        if old_socket is not None:
            close_zmq_socket(old_socket)

    def remove_instance(self, instance_id: str) -> None:
        """Close and forget an instance's command socket. No-op if unknown.

        Args:
            instance_id: The departing mp server's id.
        """
        socket = self._sockets.pop(instance_id, None)
        if socket is not None:
            close_zmq_socket(socket)
        self._reach.pop(instance_id, None)
        self._locks.pop(instance_id, None)

    def close(self) -> None:
        """Close the bound sockets and every command socket."""
        for instance_id in list(self._sockets):
            self.remove_instance(instance_id)
        close_zmq_socket(self._request_socket)
        close_zmq_socket(self._push_socket)


class ZmqClientTransport(ClientTransport):
    """ZMQ mp-server-side transport (synchronous).

    Args:
        zmq_context: A synchronous ZMQ context.
        request_url: Coordinator request/reply ROUTER address.
        push_url: Coordinator fire-and-forget PULL address.
        control_port: Port to bind the control REP socket on.
    """

    def __init__(
        self,
        zmq_context: zmq.Context,
        request_url: str,
        push_url: str,
        control_port: int,
    ) -> None:
        """Store endpoints; sockets are opened lazily per operation."""
        self._zmq_context = zmq_context
        self._request_url = request_url
        self._push_url = push_url
        self._control_port = control_port

    def request(self, payload: bytes, timeout_ms: int) -> bytes:
        """Send a request to the coordinator and wait for the reply.

        A fresh REQ socket per call keeps the strict REQ state machine from
        wedging across calls and is inherently thread-safe.

        Args:
            payload: Opaque, encoded request bytes.
            timeout_ms: Maximum time to wait for the reply.

        Returns:
            The coordinator's raw reply bytes.

        Raises:
            TransportError: On send/receive failure or timeout.
        """
        socket = get_zmq_socket_with_timeout(
            self._zmq_context,
            self._request_url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
            recv_timeout_ms=timeout_ms,
            send_timeout_ms=DEFAULT_SOCKET_SEND_TIMEOUT_MS,
        )
        try:
            socket.send(payload)
            return socket.recv()
        except zmq.ZMQError as e:
            raise TransportError(f"Request to {self._request_url} failed: {e}") from e
        finally:
            close_zmq_socket(socket)

    def push(self, payload: bytes) -> None:
        """Send a fire-and-forget message to the coordinator.

        Args:
            payload: Opaque, encoded message bytes.
        """
        socket = get_zmq_socket(
            self._zmq_context,
            self._push_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )
        try:
            socket.send(payload)
        except zmq.ZMQError as e:
            logger.warning("Push to %s failed: %s", self._push_url, e)
        finally:
            # Linger briefly so the message flushes before the socket closes.
            close_zmq_socket(socket, linger=200)

    def serve_commands(
        self, handler: ClientCommandHandler, should_stop: StopPredicate
    ) -> None:
        """Serve coordinator-pushed commands until ``should_stop`` is true.

        Args:
            handler: Maps a command payload to a reply payload.
            should_stop: Polled between receives; returning true ends the loop.
        """
        socket = get_zmq_socket(
            self._zmq_context,
            f"0.0.0.0:{self._control_port}",
            protocol="tcp",
            role=zmq.REP,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        socket.setsockopt(zmq.RCVTIMEO, _CONTROL_POLL_TIMEOUT_MS)
        try:
            while not should_stop():
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if should_stop():
                        break
                    logger.error("Control loop receive error: %s", e)
                    continue
                try:
                    reply = handler(payload)
                except Exception as e:
                    logger.error("Command handler failed: %s", e)
                    reply = b"error"
                socket.send(reply)
        finally:
            close_zmq_socket(socket)

    def close(self) -> None:
        """No persistent sockets to close (request/push are per-call)."""
        return None
