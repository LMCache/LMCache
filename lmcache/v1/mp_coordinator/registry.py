# SPDX-License-Identifier: Apache-2.0
"""Thread-safe registry of live mp servers known to the coordinator.

The registry is the single source of truth for fleet membership. It is mutated
by the coordinator event loop (register / deregister / heartbeat) and read by
the health-check thread (stale detection), so every access is guarded by a
lock.

The registry stores plain data only. The per-instance command socket is owned
here as an opaque handle but is never opened or closed by the registry --
socket lifecycle is the controller's responsibility and must stay on the event
loop thread (ZMQ sockets are not thread-safe).
"""

# Standard
from dataclasses import dataclass, field
import threading
import time

# Third Party
import zmq.asyncio

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class MPInstanceNode:
    """A single registered mp server.

    Attributes:
        instance_id: Globally unique identifier of the mp server.
        ip: IP address the mp server is reachable at.
        control_port: Port of the mp server's control REP socket.
        command_socket: REQ socket connected to the mp server's control REP
            socket, used by the coordinator to push commands. Owned by the
            event loop thread.
        registration_time: Wall-clock time the instance registered (for display).
        last_heartbeat_time: Monotonic-clock time of the most recent heartbeat,
            used for stale detection so an NTP step cannot skew liveness.
        metadata: Free-form string key/value pairs supplied at registration.
    """

    instance_id: str
    ip: str
    control_port: int
    command_socket: zmq.asyncio.Socket
    registration_time: float
    last_heartbeat_time: float
    metadata: dict[str, str] = field(default_factory=dict)


class InstanceRegistry:
    """Thread-safe in-memory registry of mp servers.

    All public methods acquire an internal lock, so the registry is safe to
    share between the coordinator event loop and the health-check thread.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._instances: dict[str, MPInstanceNode] = {}

    def register(self, node: MPInstanceNode) -> None:
        """Insert or replace an mp server entry.

        Args:
            node: The instance node to store. If an instance with the same
                ``instance_id`` already exists it is overwritten; the caller is
                responsible for closing any prior command socket first.
        """
        with self._lock:
            self._instances[node.instance_id] = node

    def deregister(self, instance_id: str) -> MPInstanceNode | None:
        """Remove an mp server entry and return it.

        Args:
            instance_id: Identifier of the instance to remove.

        Returns:
            The removed node, or ``None`` if no such instance was registered.
            The caller closes the returned node's command socket.
        """
        with self._lock:
            return self._instances.pop(instance_id, None)

    def get(self, instance_id: str) -> MPInstanceNode | None:
        """Return the node for an instance, or ``None`` if unknown.

        Args:
            instance_id: Identifier to look up.

        Returns:
            The matching node, or ``None``.
        """
        with self._lock:
            return self._instances.get(instance_id)

    def contains(self, instance_id: str) -> bool:
        """Report whether an instance is currently registered.

        Args:
            instance_id: Identifier to check.

        Returns:
            ``True`` if registered, otherwise ``False``.
        """
        with self._lock:
            return instance_id in self._instances

    def all_instances(self) -> list[MPInstanceNode]:
        """Return a snapshot list of all registered nodes.

        Returns:
            A new list containing every currently registered node.
        """
        with self._lock:
            return list(self._instances.values())

    def update_heartbeat(self, instance_id: str, timestamp: float) -> bool:
        """Record a heartbeat timestamp for an instance.

        Args:
            instance_id: Identifier of the instance.
            timestamp: Monotonic-clock time of the heartbeat (see
                :meth:`stale`); must come from the same clock as ``stale``.

        Returns:
            ``True`` if the instance was found and updated, ``False`` if it is
            not registered (the caller should treat this as a re-register).
        """
        with self._lock:
            node = self._instances.get(instance_id)
            if node is None:
                return False
            node.last_heartbeat_time = timestamp
            return True

    def stale(self, timeout: float) -> list[str]:
        """Return the ids of instances whose heartbeat has expired.

        Args:
            timeout: Maximum allowed seconds since the last heartbeat.

        Returns:
            A list of instance ids that have not sent a heartbeat within
            ``timeout`` seconds.
        """
        now = time.monotonic()
        with self._lock:
            return [
                instance_id
                for instance_id, node in self._instances.items()
                if now - node.last_heartbeat_time > timeout
            ]
