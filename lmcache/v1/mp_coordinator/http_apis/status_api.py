# SPDX-License-Identifier: Apache-2.0
"""One JSON snapshot of the coordinator's state, for a person debugging it.

Every section is a component's ``stats()`` (or the event source's
``status()``) -- the same snapshots the metrics read, so the two never
disagree. ``/healthz`` stays the plain liveness probe.

See ``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from dataclasses import asdict

# Third Party
from fastapi import APIRouter, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry

router = APIRouter()


@router.get("/status")
async def coordinator_status(request: Request) -> dict[str, object]:
    """Return a snapshot of the coordinator's state.

    Always 200: a section reports whatever its component has, so the page
    is most useful exactly when something is wrong.

    Args:
        request: The incoming request, carrying the coordinator context.

    Returns:
        A JSON object with one key per section:

        - ``event_source``: which source feeds the gate; for Kafka, also
          records read, partitions held, unread records and next offsets.
        - ``ingest``: the gate's fleet-wide counts, plus each server's
          stream (``streams``): incarnation, last ``seq``, losses,
          freshness.
        - ``consumers``: each consumer's deliveries, apply time and
          failures.
        - ``registry``: seconds since each registered server's heartbeat.
        - ``controllers``: whether each controller started.
        - ``background_loops``: each loop's passes, last success and its
          age, and last error. The ``checkpoint`` loop is the checkpoint's
          state; it is absent when periodic checkpoints are off.
        - ``event_loop``: how long the event loop was blocked.
    """
    ctx = get_context(request)
    return {
        "event_source": asdict(ctx.event_source.status()),
        "ingest": asdict(ctx.event_gate.stats()),
        "consumers": {
            name: asdict(stats) for name, stats in ctx.event_broadcaster.stats().items()
        },
        "registry": asdict(ctx.views.get(InstanceRegistry).stats()),
        "controllers": ctx.controller_status.stats(),
        "background_loops": {
            name: asdict(stats) for name, stats in ctx.background_loops.stats().items()
        },
        "event_loop": asdict(ctx.event_loop_monitor.stats()),
    }
