# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from fastapi import APIRouter
from prometheus_client import REGISTRY, generate_latest
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> Any:
    """
    Provide Prometheus metrics data in the standard
    exposition format.
    """
    metrics_data = generate_latest(REGISTRY)
    return PlainTextResponse(content=metrics_data, media_type="text/plain")
