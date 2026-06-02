# SPDX-License-Identifier: Apache-2.0
"""Transport boundary for the mp coordinator.

The coordinator core (dispatch, controllers, registry) speaks logical
operations on opaque ``bytes``. A transport implementation hides the wire
mechanism -- ZMQ sockets today, gRPC streams or NATS subjects later -- including
how to reach an individual mp server for server-initiated push.

Two sides, intentionally different concurrency models because they live in
different processes:

- :class:`CoordinatorTransport` is async (the coordinator runs an asyncio loop).
- :class:`ClientTransport` is synchronous (an mp server drives it from threads).
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable


class TransportError(Exception):
    """A send/receive failed at the transport layer (e.g. timeout)."""


class UnknownInstanceError(KeyError):
    """A command was addressed to an instance the transport cannot reach."""


class InboundKind(Enum):
    """Whether an inbound message expects a reply."""

    REQUEST = "request"
    PUSH = "push"


@dataclass
class Inbound:
    """A message arriving at the coordinator.

    Attributes:
        kind: ``REQUEST`` (a reply is expected) or ``PUSH`` (fire-and-forget).
        payload: The opaque, encoded message bytes.
        instance_id: The sender's id if the transport knows it, else ``None``
            (the core learns the sender by decoding the payload).
    """

    kind: InboundKind
    payload: bytes
    instance_id: str | None = None


@dataclass
class ReachInfo:
    """How the coordinator can reach an mp server for server-initiated push.

    Transport-specific: a ZMQ transport dials ``ip:control_port``; a
    stream/subject transport ignores these (the inbound connection is the reach)
    and keys off ``instance_id`` / ``metadata`` instead.

    Attributes:
        ip: The mp server's IP address.
        control_port: The mp server's control port.
        metadata: Free-form hints supplied at registration.
    """

    ip: str
    control_port: int
    metadata: dict[str, str] = field(default_factory=dict)


# Core-supplied handler: returns reply bytes for a REQUEST, None for a PUSH.
InboundHandler = Callable[[Inbound], Awaitable[bytes | None]]

# Client-side handler for a pushed command: maps a command payload to a reply.
ClientCommandHandler = Callable[[bytes], bytes]

# Predicate the client control loop polls to know when to stop serving.
StopPredicate = Callable[[], bool]


class CoordinatorTransport(ABC):
    """Coordinator (server) side of the transport."""

    @abstractmethod
    async def serve(self, handler: InboundHandler) -> None:
        """Run until cancelled, routing inbound messages to ``handler``.

        For each :class:`Inbound` of kind ``REQUEST`` the handler's returned
        bytes are sent back to the originating sender; ``PUSH`` messages get no
        reply.

        Args:
            handler: The coordinator core's dispatch callback.
        """

    @abstractmethod
    async def send_command(self, instance_id: str, payload: bytes) -> bytes:
        """Push a command to one instance and return its reply.

        Args:
            instance_id: Target mp server.
            payload: Opaque, encoded command bytes.

        Returns:
            The instance's raw reply bytes.

        Raises:
            UnknownInstanceError: If the instance is not reachable.
            TransportError: If the exchange fails; the transport repairs its
                own connection so a later command can recover.
        """

    @abstractmethod
    async def broadcast(self, payload: bytes) -> dict[str, bytes]:
        """Push a command to every reachable instance.

        Per-instance failures are skipped (logged), not raised.

        Args:
            payload: Opaque, encoded command bytes.

        Returns:
            A mapping of instance id to reply bytes for instances that replied.
        """

    @abstractmethod
    def add_instance(self, instance_id: str, reach: ReachInfo) -> None:
        """Record how to reach an instance (called on registration).

        Stream/subject transports may treat this as a no-op.

        Args:
            instance_id: The registering mp server's id.
            reach: How to reach it for push.

        Raises:
            TransportError: If a reach connection could not be established.
        """

    @abstractmethod
    def remove_instance(self, instance_id: str) -> None:
        """Forget an instance and release its reach (called on deregister).

        A no-op if the instance is unknown.

        Args:
            instance_id: The departing mp server's id.
        """

    @abstractmethod
    def close(self) -> None:
        """Close all transport resources. Idempotent."""


class ClientTransport(ABC):
    """mp-server (client) side of the transport. Synchronous, thread-driven."""

    @abstractmethod
    def request(self, payload: bytes, timeout_ms: int) -> bytes:
        """Send a request to the coordinator and wait for the reply.

        Used for register and heartbeat.

        Args:
            payload: Opaque, encoded request bytes.
            timeout_ms: Maximum time to wait for the reply.

        Returns:
            The coordinator's raw reply bytes.

        Raises:
            TransportError: On send/receive failure or timeout.
        """

    @abstractmethod
    def push(self, payload: bytes) -> None:
        """Send a fire-and-forget message to the coordinator (deregister).

        Args:
            payload: Opaque, encoded message bytes.
        """

    @abstractmethod
    def serve_commands(
        self, handler: ClientCommandHandler, should_stop: StopPredicate
    ) -> None:
        """Serve coordinator-pushed commands until ``should_stop`` is true.

        Blocks; runs on the client's control thread. Each received command is
        passed to ``handler`` and the returned bytes sent back.

        Args:
            handler: Maps a command payload to a reply payload.
            should_stop: Polled between receives; returning true ends the loop.
        """

    @abstractmethod
    def close(self) -> None:
        """Close all transport resources. Idempotent."""
