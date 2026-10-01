# SPDX-License-Identifier: Apache-2.0
"""L1 occupancy reporting for the multiprocess (MP) cache server.

Answers "how full is L1, in bytes": a flush thread probes the
``StorageManager`` every ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds and
sends one ``L1UsageMessage`` per interval. The reporter subscribes to no
events; it registers on the EventBus only so that ``EventBus.stop()``
drives its shutdown and final flush, like the other MP reporters.

This module is not re-exported from the package root so that importing
:mod:`lmcache.usage_telemetry` (done by the single-process engine path)
never pulls in :mod:`lmcache.v1.mp_observability`.
"""

# Future
from __future__ import annotations

# Standard
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
from lmcache.usage_telemetry.messages import DeploymentMode, L1UsageMessage
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


class L1UsageReporter(EventSubscriber):
    """L1 occupancy reporter for the multiprocess cache server.

    Every flush interval, sends one ``L1UsageMessage`` carrying the
    L1 pool's current used and total bytes. A failed probe reports the
    ``bytes_used=-1`` / ``capacity_bytes=0`` sentinels instead of
    skipping the interval. A final flush is sent when the owning
    EventBus stops.
    """

    def __init__(
        self,
        usage_probe: Callable[[], tuple[int, int]],
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter and start its flush thread.

        Args:
            usage_probe: Returns the L1 pool's current
                ``(used_bytes, total_bytes)``. Called once per flush on
                the flush thread; failures are swallowed (that flush
                reports the occupancy sentinels).
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._usage_probe = usage_probe
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._lock = threading.Lock()
        self._sequence_number = 0
        self._start_monotonic = time.monotonic()
        self._last_flush_monotonic = self._start_monotonic
        self._flush_thread = start_usage_flush_thread("lmcache-usage-l1", self.flush)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {}

    def shutdown(self) -> None:
        """Stop the flush thread and send a final partial-interval flush.

        Called by ``EventBus.stop()``.
        """
        self._flush_thread.stop()
        self.flush()

    @swallow_telemetry_errors
    def flush(self) -> None:
        """Send one L1 occupancy message for the elapsed interval.

        Called periodically by the internal flush thread; safe to call
        from any thread. When usage tracking is disabled nothing is
        sent. Never raises.
        """
        now = time.monotonic()
        with self._lock:
            self._sequence_number += 1
            sequence_number = self._sequence_number
            active_seconds = now - self._last_flush_monotonic
            self._last_flush_monotonic = now
        if not is_usage_tracking_enabled():
            return
        try:
            bytes_used, capacity_bytes = self._usage_probe()
        except Exception:
            logger.debug("L1 usage probe failed", exc_info=True)
            bytes_used, capacity_bytes = -1, 0
        message = L1UsageMessage(
            active_seconds=active_seconds,
            bytes_used=int(bytes_used),
            capacity_bytes=int(capacity_bytes),
            sequence_number=sequence_number,
            uptime_seconds=now - self._start_monotonic,
        )
        payload = build_usage_payload(
            message, get_usage_identity(), DeploymentMode.MP_SERVER
        )
        self._sender.send(usage_server_url(message.ENDPOINT), payload)


@swallow_telemetry_errors
def InitializeL1Usage(
    event_bus: EventBus,
    storage_manager: StorageManager,
    sender: UsageMessageSender | None = None,
) -> L1UsageReporter | None:
    """Start L1 occupancy reporting for an MP cache server.

    Registers an :class:`L1UsageReporter` on *event_bus* with a probe
    backed by ``StorageManager.get_l1_usage()``. Never blocks or raises.

    Args:
        event_bus: The server's started EventBus.
        storage_manager: The server's storage manager, probed at each
            flush for the L1 pool's occupancy.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The reporter, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None

    logger.info("Initializing L1 usage reporting.")
    reporter = L1UsageReporter(storage_manager.get_l1_usage, sender)
    event_bus.register_subscriber(reporter)
    return reporter
