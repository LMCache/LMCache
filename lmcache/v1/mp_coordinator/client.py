# SPDX-License-Identifier: Apache-2.0
"""mp-server-side client for the mp coordinator.

An mp server embeds a :class:`CoordinatorClient` to join the coordinator, send
heartbeats, and serve commands the coordinator pushes to it. The client owns the
lifecycle and threads; all wire I/O is delegated to an injected
:class:`ClientTransport`, so the client itself is transport-agnostic.

Threads:

- the calling thread runs :meth:`start` (register) and :meth:`stop` (deregister),
- the heartbeat thread periodically sends a heartbeat,
- the control thread serves pushed commands.

The command handler is the seam future functionality plugs into: it receives a
raw command payload and returns a raw reply. The default handler simply
acknowledges, which is enough for the backbone.
"""

# Standard
import threading

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.message import (
    CoordMsg,
    DeregisterMsg,
    HeartbeatMsg,
    RegisterMsg,
    RegisterRetMsg,
)
from lmcache.v1.mp_coordinator.transport import (
    ClientCommandHandler,
    ClientTransport,
    TransportError,
)
from lmcache.v1.rpc_utils import get_ip

logger = init_logger(__name__)

# Default ceiling for the blocking registration round-trip so a wrong or dead
# coordinator URL fails fast instead of hanging mp server startup forever.
_DEFAULT_REGISTER_TIMEOUT_MS = 10_000
# Extra slack added to the heartbeat interval for the heartbeat reply timeout.
_HEARTBEAT_REPLY_SLACK_MS = 5_000


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
        transport: The wire transport to the coordinator.
        control_port: Port the transport binds its control socket on; reported
            to the coordinator so it can reach this server for push.
        advertise_ip: IP the coordinator should reach back to. Defaults to the
            machine's outbound IP.
        heartbeat_interval: Seconds between heartbeats.
        register_timeout_ms: Maximum time to wait for the registration reply.
        command_handler: Callable invoked for each pushed command, returning the
            reply payload. Defaults to a fixed acknowledgement.
    """

    def __init__(
        self,
        instance_id: str,
        transport: ClientTransport,
        control_port: int,
        advertise_ip: str = "",
        heartbeat_interval: float = 5.0,
        register_timeout_ms: int = _DEFAULT_REGISTER_TIMEOUT_MS,
        command_handler: ClientCommandHandler = _default_command_handler,
    ) -> None:
        """Initialize client state; the transport opens resources lazily."""
        self.instance_id = instance_id
        self._transport = transport
        self.control_port = control_port
        self.advertise_ip = advertise_ip or get_ip()
        self.heartbeat_interval = heartbeat_interval
        self.register_timeout_ms = register_timeout_ms
        self.command_handler = command_handler

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
        """Register with the coordinator and start the background threads.

        Returns:
            The coordinator's registration reply.

        Raises:
            RuntimeError: If registration fails (timeout/transport error) or the
                coordinator returns an unexpected reply type.
        """
        try:
            raw_reply = self._transport.request(
                msgspec.msgpack.encode(self._build_register_msg()),
                self.register_timeout_ms,
            )
        except TransportError as e:
            raise RuntimeError(f"Registration failed: {e}") from e

        reply = msgspec.msgpack.decode(raw_reply, type=CoordMsg)
        if not isinstance(reply, RegisterRetMsg):
            raise RuntimeError(f"Unexpected registration reply: {type(reply).__name__}")

        self._control_thread.start()
        self._heartbeat_thread.start()
        logger.info(
            "CoordinatorClient %s registered with coordinator", self.instance_id
        )
        return reply

    def _control_loop(self) -> None:
        """Serve pushed commands until shutdown (runs on the control thread)."""
        self._transport.serve_commands(self.command_handler, self._shutdown.is_set)

    def _heartbeat_loop(self) -> None:
        """Send heartbeats until shutdown (runs on the heartbeat thread).

        Each heartbeat is an independent request, so a transient failure is
        logged and the next interval simply tries again.
        """
        encoded = msgspec.msgpack.encode(
            HeartbeatMsg(
                instance_id=self.instance_id,
                ip=self.advertise_ip,
                control_port=self.control_port,
            )
        )
        timeout_ms = int(self.heartbeat_interval * 1000) + _HEARTBEAT_REPLY_SLACK_MS
        while not self._shutdown.wait(self.heartbeat_interval):
            try:
                self._transport.request(encoded, timeout_ms)
            except TransportError as e:
                logger.warning("Heartbeat failed: %s", e)

    def stop(self) -> None:
        """Signal shutdown, deregister from the coordinator, and join threads."""
        self._shutdown.set()
        self._heartbeat_thread.join(timeout=5.0)
        self._control_thread.join(timeout=5.0)
        self._transport.push(
            msgspec.msgpack.encode(DeregisterMsg(instance_id=self.instance_id))
        )
        self._transport.close()
        logger.info("CoordinatorClient %s stopped", self.instance_id)
