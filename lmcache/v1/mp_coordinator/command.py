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

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.registry import InstanceRegistry

logger = init_logger(__name__)


class CommandSender:
    """Sends opaque command payloads to registered mp servers.

    Args:
        registry: The shared instance registry providing per-instance command
            sockets.
    """

    def __init__(self, registry: InstanceRegistry) -> None:
        """Initialize the sender with a registry reference."""
        self._registry = registry

    async def unicast(self, instance_id: str, payload: bytes) -> bytes:
        """Send a command to one mp server and return its reply.

        Args:
            instance_id: Identifier of the target mp server.
            payload: Opaque, already-encoded command bytes.

        Returns:
            The raw reply bytes from the mp server.

        Raises:
            KeyError: If the instance is not registered.
            zmq.ZMQError: If the send or receive fails (e.g. on timeout).
        """
        node = self._registry.get(instance_id)
        if node is None:
            raise KeyError(f"Instance {instance_id} is not registered")
        await node.command_socket.send(payload)
        return await node.command_socket.recv()

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
