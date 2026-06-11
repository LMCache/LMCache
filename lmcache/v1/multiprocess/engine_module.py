# SPDX-License-Identifier: Apache-2.0
"""Protocol and types for pluggable engine modules."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Protocol

# First Party
from lmcache.v1.multiprocess.protocol import RequestType

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext


class ThreadPoolType(Enum):
    """Declares which thread pool a handler should run in."""

    SYNC = auto()
    AFFINITY = auto()
    NORMAL = auto()


@dataclass
class HandlerSpec:
    """Specification for a single message queue handler.

    Args:
        request_type: The ZMQ request type this handler serves.
        handler: The callable that processes the request.
        pool: Which thread pool the handler runs in.
    """

    request_type: RequestType
    handler: Callable
    pool: ThreadPoolType


class EngineModule(Protocol):
    """Protocol for pluggable engine modules.

    Each module owns its internal state and exposes handlers
    that the compositor registers with the message queue server.
    """

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        ...

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves."""
        ...

    def report_status(self) -> dict:
        """Return module-specific status information."""
        ...

    def close(self) -> None:
        """Release resources owned by this module."""
        ...


class InstanceLivenessTarget(Protocol):
    """A module that tracks per-worker liveness and reaps stale workers.

    Implemented by transfer modules owning per-instance state keyed by
    ``instance_id``. The management module drives these from the PING
    handler and the periodic reaper; no caller touches the module's
    private state directly.
    """

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked (already reaped or never
        registered).

        Args:
            instance_id: The worker's opaque instance ID.
        """
        ...

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Evict and clean up workers that have gone silent.

        An instance that has sent at least one PING is judged against
        ``reap_timeout_s``; one that has never pinged (warming up, or dead
        before its first request) is judged against ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances;
                must be >= ``reap_timeout_s``.

        Returns:
            The instance IDs reaped during this scan.
        """
        ...

    def tracked_instance_count(self) -> int:
        """Return the number of currently tracked instances."""
        ...


class InstanceReapListener(Protocol):
    """A module notified when an instance is reaped, to drop mirrored state.

    Implemented by modules holding a second reference to a reaped instance's
    resources (e.g. blend rope mirrors of a GPU cache context).
    """

    def drop_instance_state(self, instance_id: int) -> None:
        """Release any state mirrored for the reaped instance.

        A no-op if the listener holds nothing for the instance.

        Args:
            instance_id: The reaped worker's instance ID.
        """
        ...
