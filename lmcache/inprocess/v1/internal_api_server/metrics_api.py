# SPDX-License-Identifier: Apache-2.0
"""InProcess-only metrics management endpoints for the internal API server.

The multiprocess HTTP server previously exposed ``POST /metrics/reset`` through
the shared ``common`` routers, but that handler imported the single-process
observability stack and was a no-op in MP-server mode (``PrometheusLogger`` is
never initialized there). The MP path has therefore dropped the endpoint
entirely.

The InProcess internal API server restores it locally: in this process the
engine's ``PrometheusLogger`` is live, so resetting the Prometheus metrics here
is meaningful. ``GET /metrics`` continues to come from the shared common router.
"""

# Third Party
from fastapi import APIRouter
from starlette.responses import PlainTextResponse

# First Party
from lmcache.inprocess.observability import reset_observability_metrics

router = APIRouter()


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all in-process Prometheus metrics to their initial state.

    Intended for test harnesses and benchmarks — not for production.
    """
    reset_observability_metrics()
    return PlainTextResponse(content="ok", media_type="text/plain")
