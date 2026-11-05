# SPDX-License-Identifier: Apache-2.0
"""HTTP cache-event source endpoint on the coordinator (fleet-level).

The ``/events`` surface is thin over :class:`HttpCacheEventSource`, which
passes request-ordered batches through the common ingestor to
:class:`EventGate`. It is top-level rather than under ``/directory``
because the stream feeds every consumer of it, not one of them. A
coordinator consuming cache events from Kafka instead runs no HTTP source,
and this endpoint answers 404. See
``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.ingest.http_event_source import HttpCacheEventSource
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest, CacheEventsResponse

router = APIRouter()


@router.post("/events")
async def report_cache_events(
    body: CacheEventsRequest, request: Request
) -> CacheEventsResponse:
    """Offer a list of cache-event batches to the ingest gate.

    Batches are offered in list order; per instance they must be sent in
    emission order. Duplicates and stale incarnations are dropped and
    counted, not errors.

    ``config`` batches ride the same path and reach the server-config
    registry as a consumer, so they inherit the gate's fencing and ordering.

    Args:
        body: The event batches to ingest.
        request: The FastAPI request carrying the coordinator context.

    Returns:
        Counts of applied and dropped batches.

    Raises:
        HTTPException: 404 when this coordinator consumes cache events from
            Kafka (``--event-transport kafka``), so the push door is closed.
    """
    source = get_context(request).event_source
    if not isinstance(source, HttpCacheEventSource):
        raise HTTPException(
            status_code=404,
            detail="POST /events is disabled: this coordinator consumes cache "
            "events from Kafka (--event-transport kafka)",
        )
    summary = source.ingest(body.batches)
    return CacheEventsResponse(
        applied=summary.applied,
        duplicates=summary.duplicates,
        stale=summary.stale,
    )
