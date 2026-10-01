# SPDX-License-Identifier: Apache-2.0
"""Per-connector L2 usage reporting for the multiprocess (MP) cache server.

Answers "which L2 adapter types are used, for how long, and how much
data do they move": an EventBus subscriber accumulates per-``l2_name``
traffic counters on the bus's drain thread, and a flush thread sends one
``L2ConnectorUsageMessage`` per active adapter type every
``LMCACHE_USAGE_TRACK_INTERVAL`` seconds. Adapter presence and occupancy
come from a ``StorageManager`` probe sampled at flush time, so they stay
correct across ``/reconfigure`` and runtime adapter add/remove without
any new events.

This module is not re-exported from the package root so that importing
:mod:`lmcache.usage_telemetry` (done by the single-process engine path)
never pulls in :mod:`lmcache.v1.mp_observability`.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.flush import start_usage_flush_thread
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import DeploymentMode, L2ConnectorUsageMessage
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


@dataclass(frozen=True)
class L2TypeUsage:
    """Occupancy snapshot of one L2 adapter type, aggregated per type."""

    bytes_used: int
    """Bytes held across all adapters of this type."""

    capacity_bytes: int
    """Summed capacity of the capacity-bounded adapters of this type;
    ``0`` when every adapter of the type is unbounded/unknown."""

    unbounded_adapters: int
    """Adapters of this type without a known capacity. Nonzero means
    ``bytes_used / capacity_bytes`` is not a meaningful occupancy ratio."""


@dataclass
class _TypeCounters:
    """Interval traffic counters of one L2 adapter type."""

    stored_bytes: int = 0
    store_succeeded_keys: int = 0
    store_failed_keys: int = 0
    load_submitted_keys: int = 0
    load_submitted_bytes: int = 0


class L2ConnectorUsageReporter(EventSubscriber):
    """Per-connector L2 usage reporter for the multiprocess cache server.

    Every flush interval, sends one ``L2ConnectorUsageMessage`` per L2
    adapter type that is either active (present in the usage probe) or
    had traffic this interval. Idle-but-configured types still report,
    so ``active_seconds`` measures connector usage duration even without
    load; an interval with no L2 adapters and no traffic sends nothing.
    Interval data is dropped, not retried, when a send fails. A final
    flush is sent when the owning EventBus stops.
    """

    def __init__(
        self,
        usage_probe: Callable[[], dict[str, L2TypeUsage]],
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter and start its flush thread.

        Args:
            usage_probe: Returns the occupancy snapshot of every
                currently active L2 adapter type, keyed by type name.
                Called once per flush on the flush thread; failures are
                swallowed (that flush reports occupancy sentinels).
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._usage_probe = usage_probe
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._lock = threading.Lock()
        self._counters: dict[str, _TypeCounters] = {}
        self._sequence_number = 0
        self._start_monotonic = time.monotonic()
        self._last_flush_monotonic = self._start_monotonic
        self._flush_thread = start_usage_flush_thread("lmcache-usage-l2", self.flush)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_LOAD_TASK_SUBMITTED: self._on_load_task_submitted,
        }

    def shutdown(self) -> None:
        """Stop the flush thread and send a final partial-interval flush.

        Called by ``EventBus.stop()``.
        """
        self._flush_thread.stop()
        self.flush()

    @swallow_telemetry_errors
    def flush(self) -> None:
        """Send one message per active or trafficked adapter type.

        Called periodically by the internal flush thread; safe to call
        from any thread. When usage tracking is disabled the counters
        are dropped without sending. Never raises.
        """
        now = time.monotonic()
        with self._lock:
            counters = self._counters
            self._counters = {}
            self._sequence_number += 1
            sequence_number = self._sequence_number
            active_seconds = now - self._last_flush_monotonic
            self._last_flush_monotonic = now
        if not is_usage_tracking_enabled():
            return
        try:
            usage_by_type = self._usage_probe()
        except Exception:
            logger.debug("L2 usage probe failed", exc_info=True)
            usage_by_type = {}
        uptime_seconds = now - self._start_monotonic
        identity = get_usage_identity()
        for l2_name in sorted(usage_by_type.keys() | counters.keys()):
            type_counters = counters.get(l2_name, _TypeCounters())
            # A type with traffic but absent from the probe was removed
            # mid-interval; its occupancy is unknown.
            usage = usage_by_type.get(l2_name, L2TypeUsage(-1, 0, 0))
            message = L2ConnectorUsageMessage(
                l2_name=l2_name,
                active_seconds=active_seconds,
                interval_stored_bytes=type_counters.stored_bytes,
                interval_store_succeeded_keys=type_counters.store_succeeded_keys,
                interval_store_failed_keys=type_counters.store_failed_keys,
                interval_load_submitted_keys=type_counters.load_submitted_keys,
                interval_load_submitted_bytes=type_counters.load_submitted_bytes,
                bytes_used=usage.bytes_used,
                capacity_bytes=usage.capacity_bytes,
                unbounded_adapters=usage.unbounded_adapters,
                sequence_number=sequence_number,
                uptime_seconds=uptime_seconds,
            )
            payload = build_usage_payload(message, identity, DeploymentMode.MP_SERVER)
            self._sender.send(usage_server_url(message.ENDPOINT), payload)

    @swallow_telemetry_errors
    def _on_store_completed(self, event: Event) -> None:
        metadata = event.metadata
        with self._lock:
            counters = self._counters.setdefault(
                str(metadata["l2_name"]), _TypeCounters()
            )
            counters.stored_bytes += int(metadata["bytes_transferred"])
            counters.store_succeeded_keys += int(metadata["succeeded_count"])
            counters.store_failed_keys += int(metadata["failed_count"])

    @swallow_telemetry_errors
    def _on_load_task_submitted(self, event: Event) -> None:
        metadata = event.metadata
        with self._lock:
            counters = self._counters.setdefault(
                str(metadata["l2_name"]), _TypeCounters()
            )
            counters.load_submitted_keys += int(metadata["key_count"])
            counters.load_submitted_bytes += int(metadata["total_bytes"])


@swallow_telemetry_errors
def InitializeL2ConnectorUsage(
    event_bus: EventBus,
    storage_manager: StorageManager,
    sender: UsageMessageSender | None = None,
) -> L2ConnectorUsageReporter | None:
    """Start per-connector L2 usage reporting for an MP cache server.

    Registers an :class:`L2ConnectorUsageReporter` on *event_bus* with an
    occupancy probe backed by
    ``StorageManager.get_l2_usages_by_type()``. Never blocks or raises.

    Args:
        event_bus: The server's started EventBus.
        storage_manager: The server's storage manager, probed at each
            flush for active adapter types and their occupancy.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The reporter, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None

    def usage_probe() -> dict[str, L2TypeUsage]:
        summary: dict[str, L2TypeUsage] = {}
        for type_name, usages in storage_manager.get_l2_usages_by_type().items():
            summary[type_name] = L2TypeUsage(
                bytes_used=sum(int(usage.total_bytes_used) for usage in usages),
                capacity_bytes=sum(
                    int(usage.total_capacity_bytes)
                    for usage in usages
                    if usage.total_capacity_bytes > 0
                ),
                unbounded_adapters=sum(
                    1 for usage in usages if usage.total_capacity_bytes <= 0
                ),
            )
        return summary

    logger.info("Initializing L2 connector usage reporting.")
    reporter = L2ConnectorUsageReporter(usage_probe, sender)
    event_bus.register_subscriber(reporter)
    return reporter
