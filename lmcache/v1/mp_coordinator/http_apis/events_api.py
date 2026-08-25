# SPDX-License-Identifier: Apache-2.0
"""Cache-event ingest endpoint on the coordinator (fleet-level).

The ``/events`` surface, thin over the :class:`EventGate`. Top-level
rather than under ``/directory`` because the stream feeds every consumer
of it, not one of them. See
``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Third Party
from fastapi import APIRouter, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.ingest.event_gate import IngestResult
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

    Returns:
        Counts of applied and dropped batches.
    """
    event_gate = get_context(request).event_gate
    response = CacheEventsResponse()
    for batch in body.batches:
        result = event_gate.ingest(batch)
        if result == IngestResult.ADMITTED:
            response.applied += 1
        elif result == IngestResult.DUPLICATE:
            response.duplicates += 1
        else:
            response.stale += 1
    return response
