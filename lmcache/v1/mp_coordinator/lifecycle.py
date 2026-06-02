# SPDX-License-Identifier: Apache-2.0
"""In-process lifecycle hooks for instance join/leave events.

This is a minimal observer registry. It exists so that controllers can
react to an mp server joining or leaving without the registration controller
importing them, keeping the dependency graph one-directional.

Contract for subscribers: callbacks run inline on the coordinator event loop.
They MUST NOT block the loop -- any heavy work (network pushes, storage reads)
must be scheduled onto a separate task so the triggering operation (e.g. a
registration reply) is not delayed.
"""

# Standard
from typing import Callable

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# A lifecycle callback receives the affected instance id.
LifecycleCallback = Callable[[str], None]


class LifecycleHooks:
    """Registry of join/leave callbacks fired by the registration controller."""

    def __init__(self) -> None:
        """Initialize with no subscribers."""
        self._on_join: list[LifecycleCallback] = []
        self._on_leave: list[LifecycleCallback] = []

    def on_join(self, callback: LifecycleCallback) -> None:
        """Subscribe a callback to fire when an mp server joins.

        Args:
            callback: A callable invoked with the joining instance id.
        """
        self._on_join.append(callback)

    def on_leave(self, callback: LifecycleCallback) -> None:
        """Subscribe a callback to fire when an mp server leaves.

        Args:
            callback: A callable invoked with the departing instance id.
        """
        self._on_leave.append(callback)

    def fire_join(self, instance_id: str) -> None:
        """Invoke every join callback for an instance.

        A failing callback is logged and does not prevent the remaining
        callbacks from running.

        Args:
            instance_id: Identifier of the joining instance.
        """
        for callback in self._on_join:
            try:
                callback(instance_id)
            except Exception as e:
                logger.error("on_join callback failed for %s: %s", instance_id, e)

    def fire_leave(self, instance_id: str) -> None:
        """Invoke every leave callback for an instance.

        A failing callback is logged and does not prevent the remaining
        callbacks from running.

        Args:
            instance_id: Identifier of the departing instance.
        """
        for callback in self._on_leave:
            try:
                callback(instance_id)
            except Exception as e:
                logger.error("on_leave callback failed for %s: %s", instance_id, e)
