# SPDX-License-Identifier: Apache-2.0
"""HTTP push implementation of the coordinator cache-event source."""

# First Party
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.ingest.event_gate import (
    CacheEventIngestSummary,
    EventGate,
)
from lmcache.v1.mp_coordinator.ingest.event_source import (
    CacheEventSource,
    CacheEventSourceStatus,
    EventReplayCapability,
)


class HttpCacheEventSource(CacheEventSource):
    """Non-durable HTTP push source for ``POST /events``.

    HTTP owns no background resource and cannot seek or replay. Delivery
    failures before the coordinator accepts a request therefore remain visible
    only as sequence gaps at the gate.

    Args:
        event_gate: Admission authority for pushed event batches.
    """

    def __init__(self, event_gate: EventGate) -> None:
        self._event_gate = event_gate

    async def start(self) -> None:
        """Start the source.

        HTTP requests are served by FastAPI, so this implementation has no
        source-owned resource to start.
        """

    async def stop(self) -> None:
        """Stop the source.

        HTTP requests are served by FastAPI, so this implementation has no
        source-owned resource to stop.
        """

    def status(self) -> CacheEventSourceStatus:
        """Return the HTTP source's non-replayable status.

        Returns:
            HTTP source identity with replay capability set to ``NONE``.
        """
        return CacheEventSourceStatus(
            source_name="http",
            replay_capability=EventReplayCapability.NONE,
        )

    def ingest(self, batches: list[CacheEventBatch]) -> CacheEventIngestSummary:
        """Deliver one HTTP request's batches to the event gate.

        Args:
            batches: Event batches in request order.

        Returns:
            Counts of admitted, duplicate, and stale batches.
        """
        return self._event_gate.ingest_batches(batches)
