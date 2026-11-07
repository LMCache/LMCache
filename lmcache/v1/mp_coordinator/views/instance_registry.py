# SPDX-License-Identifier: Apache-2.0
"""Thread-safe registry of live mp servers known to the coordinator.

The single source of truth for fleet membership, and a view like any
other -- whoever needs it asks the view registry rather than being handed
a copy. Unlike the other views it is fed by the ``/instances`` endpoints
and the health sweep, not by the cache-event stream: being shared is what
makes it a view, not what writes it.

It is mutated and read on the coordinator event loop (register / deregister
/ heartbeat / health-check eviction); access is lock-guarded to stay
correct.

The registry stores plain membership data only -- ip, http_port, heartbeat
timestamps, metadata. How to reach an instance for push is derived from its
``ip`` + ``http_port`` by whichever router needs to call it.
"""

# Standard
from dataclasses import dataclass, field
import random
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source
from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)


@dataclass
class MPInstance:
    """A single registered mp server.

    Attributes:
        instance_id: Globally unique identifier of the mp server.
        ip: IP address the mp server's HTTP server is reachable at.
        http_port: Port of the mp server's HTTP server, which the coordinator
            calls to push work to this instance.
        registration_time: Wall-clock time the instance registered (for display).
        last_heartbeat_time: Monotonic-clock time of the most recent heartbeat,
            used for stale detection so an NTP step cannot skew liveness.
        metadata: Free-form string key/value pairs supplied at registration.
        p2p_advertised_url: URL this instance advertises for peer-to-peer
            transfers. Empty when the instance does not participate in P2P.
        mq_port: Port of the instance's ZMQ message-queue server that P2P peers
            send lookup/unlock RPCs to, reachable at this instance's ``ip``. 0
            when P2P is disabled.
    """

    instance_id: str
    ip: str
    http_port: int
    registration_time: float
    last_heartbeat_time: float
    metadata: dict[str, str] = field(default_factory=dict)
    p2p_advertised_url: str = ""
    mq_port: int = 0


@dataclass(frozen=True)
class InstanceRegistryStats:
    """Fleet membership and heartbeat freshness.

    Attributes:
        servers_registered: Number of registered mp servers.
        seconds_since_heartbeat: Seconds since each registered server's
            last heartbeat, keyed by instance id.
        membership_changes: Changes since this process started, by kind:
            ``registered`` (a new id), ``re_registered`` (a known id
            registered again), ``deregistered`` (removed on request),
            ``timed_out`` (removed by :meth:`InstanceRegistry.remove_stale`)
            and ``unknown_heartbeat`` (a heartbeat from an id not
            registered). Every kind is present, from 0.
    """

    servers_registered: int
    seconds_since_heartbeat: dict[str, float]
    membership_changes: dict[str, int]


class InstanceRegistry(View):
    """Thread-safe in-memory registry of mp servers.

    All public methods acquire an internal lock, so the registry stays
    consistent under concurrent access.

    Holds nothing durable: membership rebuilds from the heartbeats that
    arrive after a restart, so a captured copy would only resurrect
    instances that may be gone.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._instances: dict[str, MPInstance] = {}
        self._membership_changes = dict.fromkeys(
            (
                "registered",
                "re_registered",
                "deregistered",
                "timed_out",
                "unknown_heartbeat",
            ),
            0,
        )
        register_metric_source(self)

    def register(self, instance: MPInstance) -> bool:
        """Insert or replace an mp server entry atomically.

        Args:
            instance: The instance to store. If one with the same
                ``instance_id`` already exists it is overwritten.

        Returns:
            ``True`` if an instance with the same id already existed (i.e. this
            was a re-registration), else ``False``. The check and the write
            happen under one lock, so concurrent registrations of the same id
            cannot both report ``False``.
        """
        with self._lock:
            existed = instance.instance_id in self._instances
            self._instances[instance.instance_id] = instance
            self._membership_changes["re_registered" if existed else "registered"] += 1
            return existed

    def deregister(self, instance_id: str) -> MPInstance | None:
        """Remove an mp server entry and return it.

        Args:
            instance_id: Identifier of the instance to remove.

        Returns:
            The removed instance, or ``None`` if no such instance was
            registered.
        """
        with self._lock:
            removed = self._instances.pop(instance_id, None)
            if removed is not None:
                self._membership_changes["deregistered"] += 1
            return removed

    def get(self, instance_id: str) -> MPInstance | None:
        """Return the instance with the given id, or ``None`` if unknown.

        Args:
            instance_id: Identifier to look up.

        Returns:
            The matching instance, or ``None``.
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

    def all_instances(self) -> list[MPInstance]:
        """Return a snapshot list of all registered instances.

        Returns:
            A new list containing every currently registered instance.
        """
        with self._lock:
            return list(self._instances.values())

    def stats(self) -> InstanceRegistryStats:
        """Return a snapshot of fleet membership; see
        :class:`InstanceRegistryStats`."""
        now = time.monotonic()
        with self._lock:
            return InstanceRegistryStats(
                servers_registered=len(self._instances),
                seconds_since_heartbeat={
                    instance_id: now - instance.last_heartbeat_time
                    for instance_id, instance in self._instances.items()
                },
                membership_changes=dict(self._membership_changes),
            )

    def random_instance(self) -> "MPInstance | None":
        """Return a uniformly random registered instance, or ``None``
        when the registry is empty."""
        with self._lock:
            instances = list(self._instances.values())
        if not instances:
            return None
        return random.choice(instances)

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
            instance = self._instances.get(instance_id)
            if instance is None:
                self._membership_changes["unknown_heartbeat"] += 1
                return False
            instance.last_heartbeat_time = timestamp
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
                for instance_id, instance in self._instances.items()
                if now - instance.last_heartbeat_time > timeout
            ]

    def remove_stale(self, timeout: float) -> list[str]:
        """Remove every instance whose heartbeat has expired.

        Finding and removing happen under one lock, so an instance that
        registers again or heartbeats meanwhile is never removed.

        Args:
            timeout: Maximum allowed seconds since the last heartbeat.

        Returns:
            The ids of the instances removed.
        """
        now = time.monotonic()
        with self._lock:
            expired_instance_ids = [
                instance_id
                for instance_id, instance in self._instances.items()
                if now - instance.last_heartbeat_time > timeout
            ]
            for instance_id in expired_instance_ids:
                del self._instances[instance_id]
            self._membership_changes["timed_out"] += len(expired_instance_ids)
            return expired_instance_ids
