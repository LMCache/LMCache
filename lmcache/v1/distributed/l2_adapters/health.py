# SPDX-License-Identifier: Apache-2.0
"""Types shared by L2 adapters and the L2 health monitor.

An adapter that can tell whether its backend is alive reports it with
:class:`HealthProbe`; the monitor (``lmcache.v1.distributed.l2_health_monitor``)
turns those reports into a :class:`BackendState` per adapter.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import enum


class ProbeStatus(enum.Enum):
    """Outcome of one :meth:`L2AdapterInterface.probe_health` call."""

    OK = "ok"
    """The backend answered and the adapter's own connection to it works."""

    UNREACHABLE = "unreachable"
    """The backend did not answer. Nothing the adapter can do will help until
    the backend is back."""

    RECONNECT_NEEDED = "reconnect_needed"
    """The backend answered, but the adapter's connection to it is broken
    (for example, its worker threads died, or a stale socket or file handle
    is cached). Calling :meth:`L2AdapterInterface.reconnect` can fix it."""


@dataclass(frozen=True)
class HealthProbe:
    """Result of probing one adapter's backend."""

    status: ProbeStatus
    """What the probe found."""

    reason: str = ""
    """Short human-readable cause, shown in ``/status`` and logs. Empty on
    ``ProbeStatus.OK``."""


class BackendState(enum.Enum):
    """The monitor's view of one adapter's backend."""

    HEALTHY = "healthy"
    """New work may be routed to the adapter."""

    UNAVAILABLE = "unavailable"
    """The backend is down or the adapter's connection is broken. New work is
    not routed to the adapter and its in-flight lookups are abandoned."""

    RECOVERING = "recovering"
    """:meth:`L2AdapterInterface.reconnect` is running. New work is not routed
    to the adapter."""


class L2ReconnectError(RuntimeError):
    """:meth:`L2AdapterInterface.reconnect` could not restore the connection.

    This is an expected outcome while the backend is still down; the monitor
    logs it as a warning and retries on a later cycle.
    """
