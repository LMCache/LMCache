# SPDX-License-Identifier: Apache-2.0
"""Basic-info endpoints: liveness, health, status, and version.

These are the small, dependency-light routes used to inspect that the
server is up and what it is running:

- ``GET /`` — static liveness payload (does not touch the engine).
- ``GET /healthcheck`` — Kubernetes liveness/readiness probe.
- ``GET /status`` — detailed internal state of all MP components.
- ``GET /version``, ``/lmc_version``, ``/commit_id`` — version descriptors.
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache import utils

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    """
    Basic liveness check endpoint.
    Returns:
        dict: A dictionary containing the status and service name.
    """
    return {"status": "ok", "service": "LMCache HTTP API"}


@router.get("/healthcheck")
async def healthcheck(request: Request) -> Any:
    """
    Health check endpoint for k8s liveness/readiness probes.

    Checks:
        - HTTP server is alive (implicit: if you get a response)
        - Cache engine is alive
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "engine not initialized",
            },
        )

    return {"status": "healthy"}


@router.get("/status")
async def status(request: Request) -> Any:
    """
    Detailed status endpoint for inspecting internal state
    of all MP components (L1 cache, L2 adapters, controllers,
    sessions).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.report_status()


@router.get("/lmc_version")
async def get_lmc_version():
    return utils.VERSION


@router.get("/commit_id")
async def get_commit_id():
    return utils.COMMIT_ID


@router.get("/version")
async def get_version():
    return utils.get_version()
