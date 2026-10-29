# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter
from starlette.responses import PlainTextResponse

# First Party
from lmcache.inprocess.observability import reset_observability_metrics

router = APIRouter()


@router.post("/metrics/reset")
async def reset_metrics() -> PlainTextResponse:
    """Reset InProcess Prometheus metrics to their initial state."""
    reset_observability_metrics()
    return PlainTextResponse(content="ok", media_type="text/plain")
