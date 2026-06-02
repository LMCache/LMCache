# SPDX-License-Identifier: Apache-2.0
"""Coordinator -> mp server command channel (server-initiated push).

The coordinator pushes commands to mp servers over the per-instance REQ socket
stored in the registry (connected to each mp server's control REP socket). This
is the channel future controllers use for quota broadcast, KV-op fan-out, etc.

The sender is deliberately payload-agnostic: it moves opaque ``bytes`` and
returns the raw reply. Each controller encodes and decodes its own message
types around it, so new functionality needs no change here.

All methods MUST be awaited on the coordinator event loop thread, because the
command sockets they use are ZMQ sockets and are not thread-safe.
"""

# Standard
import asyncio

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    close_zmq_socket,
    get_zmq_socket_with_timeout,
)

logger = init_logger(__name__)


def open_command_socket(
    zmq_context: zmq.asyncio.Context, ip: str, control_port: int
) -> zmq.asyncio.Socket:
    """Open a REQ command socket connected to an mp server's control REP socket.

    Used both when an instance registers and when the channel is repaired after
    a wedged exchange. A plain REQ socket (no ``REQ_CORRELATE``) is used so the
    wire framing stays compatible with the coordinator's ROUTER parsing.

    Args:
        zmq_context: The shared async ZMQ context.
        ip: The mp server's IP address.
        control_port: The mp server's control REP port.

    Returns:
        A connected REQ socket with send/recv timeouts set.
    """
    return get_zmq_socket_with_timeout(
        zmq_context,
        f"{ip}:{control_port}",
        protocol="tcp",
        role=zmq.REQ,  # type: ignore[attr-defined]
        bind_or_connect="connect",
        recv_timeout_ms=DEFAULT_SOCKET_RECV_TIMEOUT_MS,
        send_timeout_ms=DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    )


class CommandSender:
    """Sends opaque command payloads to registered mp servers.

    Owns repair of its own command sockets: it holds the ZMQ context so a
    wedged REQ socket can be rebuilt in place, without depending on any
    controller.

    Args:
        registry: The shared instance registry providing per-instance command
            sockets.
        zmq_context: The shared async ZMQ context, used to rebuild a command
            socket after a wedged exchange.
    """

    def __init__(
        self, registry: InstanceRegistry, zmq_context: zmq.asyncio.Context
    ) -> None:
        """Initialize the sender with a registry and ZMQ context reference."""
        self._registry = registry
        self._zmq_context = zmq_context
        # One lock per instance serializes access to its REQ command socket: a
        # REQ socket has a strict send/recv state machine and cannot be shared
        # by overlapping request/reply pairs. Entries are created lazily.
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, instance_id: str) -> asyncio.Lock:
        """Return (creating if needed) the send lock for an instance.

        Args:
            instance_id: Identifier of the target mp server.

        Returns:
            The per-instance lock. Safe because all calls run on the single
            event loop thread.
        """
        lock = self._locks.get(instance_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[instance_id] = lock
        return lock

    async def unicast(self, instance_id: str, payload: bytes) -> bytes:
        """Send a command to one mp server and return its reply.

        The per-instance lock guarantees one request/reply pair at a time on the
        REQ socket. If an exchange fails (e.g. a recv timeout) the REQ socket is
        left in a state that rejects the next send, so it is rebuilt in place
        before the error propagates -- the following command to this instance
        recovers instead of erroring forever.

        Args:
            instance_id: Identifier of the target mp server.
            payload: Opaque, already-encoded command bytes.

        Returns:
            The raw reply bytes from the mp server.

        Raises:
            KeyError: If the instance is not registered.
            zmq.ZMQError: If the send or receive fails (e.g. on timeout). The
                socket is rebuilt before the error is raised.
        """
        node = self._registry.get(instance_id)
        if node is None:
            raise KeyError(f"Instance {instance_id} is not registered")
        async with self._lock_for(instance_id):
            try:
                await node.command_socket.send(payload)
                return await node.command_socket.recv()
            except zmq.ZMQError:
                close_zmq_socket(node.command_socket)
                node.command_socket = open_command_socket(
                    self._zmq_context, node.ip, node.control_port
                )
                raise

    async def broadcast(self, payload: bytes) -> dict[str, bytes]:
        """Send a command to every registered mp server.

        Failures to an individual instance are logged and that instance is
        omitted from the result rather than failing the whole broadcast.

        Args:
            payload: Opaque, already-encoded command bytes.

        Returns:
            A mapping of instance id to reply bytes for every instance that
            replied successfully.
        """
        instances = self._registry.all_instances()
        if not instances:
            return {}

        async def _send_one(instance_id: str) -> tuple[str, bytes] | None:
            try:
                reply = await self.unicast(instance_id, payload)
                return instance_id, reply
            except (KeyError, zmq.ZMQError) as e:
                logger.warning("Broadcast to instance %s failed: %s", instance_id, e)
                return None

        results = await asyncio.gather(
            *(_send_one(node.instance_id) for node in instances)
        )
        return {item[0]: item[1] for item in results if item is not None}
