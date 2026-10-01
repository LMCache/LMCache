# SPDX-License-Identifier: Apache-2.0
"""Prometheus scrape endpoint for the MP coordinator."""

# Third Party
from fastapi import APIRouter, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Return Prometheus metrics when the coordinator uses pull mode.

    Args:
        request: Incoming request carrying the coordinator configuration.

    Returns:
        The current Prometheus registry exposition.

    Raises:
        HTTPException: If metrics are disabled or configured for OTLP push.
    """
    config: MPCoordinatorConfig = request.app.state.config
    if not config.metrics_enabled or config.otlp_endpoint is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return Response(
        content=generate_latest(REGISTRY),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )
