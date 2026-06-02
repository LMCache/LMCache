# SPDX-License-Identifier: Apache-2.0
"""mp-server-side client for the mp coordinator.

An mp server embeds a :class:`CoordinatorClient` to join the coordinator, send
heartbeats, and serve commands the coordinator pushes to it. The client is
synchronous and thread-based (an mp server is its own process); each ZMQ socket
is confined to a single thread to respect ZMQ's threading rules:

- registration REQ socket: used once on the calling thread in :meth:`start`.
- heartbeat REQ socket: used only by the heartbeat thread.
- control REP socket: used only by the control thread.
- deregister PUSH socket: used once on the calling thread in :meth:`stop`.

The command handler is the seam future functionality plugs into: it receives a
raw command payload and returns a raw reply. The default handler simply
acknowledges, which is enough for the backbone.
"""

# Standard
from typing import Callable
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.message import (
    CoordMsg,
    DeregisterMsg,
    HeartbeatMsg,
    RegisterMsg,
    RegisterRetMsg,
)
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    close_zmq_socket,
    get_ip,
    get_zmq_socket,
    get_zmq_socket_with_timeout,
)

logger = init_logger(__name__)

# A command handler maps a raw command payload to a raw reply payload.
CommandHandler = Callable[[bytes], bytes]

_CONTROL_POLL_TIMEOUT_MS = 500
# Default ceiling for the blocking registration round-trip so a wrong or dead
# coordinator URL fails fast instead of hanging mp server startup forever.
_DEFAULT_REGISTER_TIMEOUT_MS = 10_000


def _default_command_handler(payload: bytes) -> bytes:
    """Acknowledge any command without acting on it.

    Args:
        payload: The raw command payload (ignored).

    Returns:
        A fixed acknowledgement byte string.
    """
    return b"ack"


class CoordinatorClient:
    """Connects an mp server to the mp coordinator.

    Args:
        instance_id: Globally unique identifier for this mp server.
        reply_url: Coordinator ROUTER address for request/reply (register).
        heartbeat_url: Coordinator ROUTER address for heartbeats.
        pull_url: Coordinator PULL address for fire-and-forget (deregister).
        control_port: Port this client binds its control REP socket on.
        heartbeat_interval: Seconds between heartbeats.
        advertise_ip: IP the coordinator should connect back to. Defaults to the
            machine's outbound IP.
        command_handler: Callable invoked for each pushed command, returning the
            reply payload. Defaults to a fixed acknowledgement.
        register_timeout_ms: Maximum time to wait for the registration reply
            before giving up, so a wrong or unreachable coordinator URL fails
            fast instead of hanging startup.
    """

    def __init__(
        self,
        instance_id: str,
        reply_url: str,
        heartbeat_url: str,
        pull_url: str,
        control_port: int,
        heartbeat_interval: float = 5.0,
        advertise_ip: str = "",
        command_handler: CommandHandler = _default_command_handler,
        register_timeout_ms: int = _DEFAULT_REGISTER_TIMEOUT_MS,
    ) -> None:
        """Initialize client state; no sockets are opened until ``start``."""
        self.instance_id = instance_id
        self.reply_url = reply_url
        self.heartbeat_url = heartbeat_url
        self.pull_url = pull_url
        self.control_port = control_port
        self.heartbeat_interval = heartbeat_interval
        self.advertise_ip = advertise_ip or get_ip()
        self.command_handler = command_handler
        self.register_timeout_ms = register_timeout_ms

        self._context = zmq.Context.instance()
        self._control_socket: zmq.Socket | None = None
        self._shutdown = threading.Event()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name=f"coord-client-control-{instance_id}",
            daemon=True,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"coord-client-heartbeat-{instance_id}",
            daemon=True,
        )

    def _build_register_msg(self) -> RegisterMsg:
        """Construct the registration message describing this mp server.

        Returns:
            A populated :class:`RegisterMsg`.
        """
        return RegisterMsg(
            instance_id=self.instance_id,
            ip=self.advertise_ip,
            control_port=self.control_port,
        )

    def start(self) -> RegisterRetMsg:
        """Bind the control socket, register, and start background threads.

        Returns:
            The coordinator's registration reply.

        Raises:
            RuntimeError: If registration times out, errors, or the coordinator
                returns an unexpected reply type.
        """
        self._control_socket = get_zmq_socket(
            self._context,
            f"0.0.0.0:{self.control_port}",
            protocol="tcp",
            role=zmq.REP,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        self._control_socket.setsockopt(zmq.RCVTIMEO, _CONTROL_POLL_TIMEOUT_MS)

        register_socket = get_zmq_socket_with_timeout(
            self._context,
            self.reply_url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
            recv_timeout_ms=self.register_timeout_ms,
            send_timeout_ms=DEFAULT_SOCKET_SEND_TIMEOUT_MS,
        )
        try:
            register_socket.send(msgspec.msgpack.encode(self._build_register_msg()))
            raw_reply = register_socket.recv()
        except zmq.ZMQError as e:
            close_zmq_socket(self._control_socket)
            self._control_socket = None
            raise RuntimeError(
                f"Registration with coordinator at {self.reply_url} failed: {e}"
            ) from e
        finally:
            close_zmq_socket(register_socket)

        reply = msgspec.msgpack.decode(raw_reply, type=CoordMsg)
        if not isinstance(reply, RegisterRetMsg):
            close_zmq_socket(self._control_socket)
            self._control_socket = None
            raise RuntimeError(f"Unexpected registration reply: {type(reply).__name__}")

        self._control_thread.start()
        self._heartbeat_thread.start()
        logger.info(
            "CoordinatorClient %s registered with coordinator", self.instance_id
        )
        return reply

    def _control_loop(self) -> None:
        """Serve pushed commands on the control REP socket until shutdown."""
        control_socket = self._control_socket
        if control_socket is None:
            logger.error("Control loop started without a bound socket")
            return
        while not self._shutdown.is_set():
            try:
                payload = control_socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if self._shutdown.is_set():
                    break
                logger.error("Control loop receive error: %s", e)
                continue
            try:
                reply = self.command_handler(payload)
            except Exception as e:
                logger.error("Command handler failed: %s", e)
                reply = b"error"
            control_socket.send(reply)

    def _make_heartbeat_socket(self) -> zmq.Socket:
        """Open a fresh REQ socket for sending heartbeats.

        Returns:
            A connected REQ socket with a receive timeout set.
        """
        socket = get_zmq_socket(
            self._context,
            self.heartbeat_url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )
        socket.setsockopt(zmq.RCVTIMEO, int(self.heartbeat_interval * 1000) + 5000)
        return socket

    def _heartbeat_once(self, socket: zmq.Socket, encoded: bytes) -> zmq.Socket:
        """Send one heartbeat, rebuilding the socket if the exchange fails.

        A REQ socket has a strict send/recv state machine: after a recv timeout
        it refuses the next send. So on any failure the socket is closed and a
        fresh one returned, otherwise a single transient miss would permanently
        stop heartbeats (and the coordinator would evict this instance).

        Args:
            socket: The current heartbeat REQ socket.
            encoded: The pre-encoded heartbeat payload.

        Returns:
            The same socket on success, or a fresh socket if the exchange
            failed and the old one was rebuilt.
        """
        try:
            socket.send(encoded)
            socket.recv()
            return socket
        except zmq.ZMQError as e:
            logger.warning("Heartbeat failed, rebuilding socket: %s", e)
            close_zmq_socket(socket)
            return self._make_heartbeat_socket()

    def _heartbeat_loop(self) -> None:
        """Send heartbeats to the coordinator until shutdown.

        A plain REQ socket is used so the wire framing matches the coordinator's
        ROUTER; :meth:`_heartbeat_once` rebuilds it on failure so a transient
        miss does not permanently stop heartbeats.
        """
        socket = self._make_heartbeat_socket()
        heartbeat = HeartbeatMsg(
            instance_id=self.instance_id,
            ip=self.advertise_ip,
            control_port=self.control_port,
        )
        encoded = msgspec.msgpack.encode(heartbeat)
        try:
            while not self._shutdown.wait(self.heartbeat_interval):
                socket = self._heartbeat_once(socket, encoded)
        finally:
            close_zmq_socket(socket)

    def stop(self) -> None:
        """Signal shutdown, deregister from the coordinator, and join threads."""
        self._shutdown.set()
        self._heartbeat_thread.join(timeout=5.0)
        self._control_thread.join(timeout=5.0)

        push_socket = get_zmq_socket(
            self._context,
            self.pull_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )
        try:
            push_socket.send(
                msgspec.msgpack.encode(DeregisterMsg(instance_id=self.instance_id))
            )
        except zmq.ZMQError as e:
            logger.warning("Deregister send failed: %s", e)
        finally:
            close_zmq_socket(push_socket, linger=200)

        if self._control_socket is not None:
            close_zmq_socket(self._control_socket)
            self._control_socket = None
        logger.info("CoordinatorClient %s stopped", self.instance_id)
