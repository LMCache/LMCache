# SPDX-License-Identifier: Apache-2.0
"""Basic-info endpoints: liveness, health, status, and version.

These are the small, dependency-light routes used to inspect that the
server is up and what it is running:

- ``GET /`` — static liveness payload (does not touch the engine).
- ``GET /healthcheck`` — Kubernetes liveness/readiness probe.
- ``GET /status`` — detailed internal state of all MP components.
- ``GET /status/memory-pressure`` — local L1/per-adapter L2 capacity pressure.
- ``GET /version``, ``/lmc_version``, ``/commit_id`` — version descriptors,
  re-exposed from the shared ``internal_api_server.vllm`` version router.
"""

# Standard
from http import HTTPStatus
from typing import Any
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.internal_api_server.vllm.version_api import router as _version_router
from lmcache.v1.multiprocess.http_apis.dependencies import get_context
from lmcache.v1.multiprocess.memory_pressure import (
    InstanceMemoryPressureReport,
    MemoryPressureUnavailable,
)

router = APIRouter()

# Re-expose the shared version routes (/version, /lmc_version, /commit_id)
# as part of the basic-info group.
router.include_router(_version_router)


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


@router.get(
    "/status/memory-pressure",
    response_model=InstanceMemoryPressureReport,
)
async def memory_pressure(request: Request) -> InstanceMemoryPressureReport:
    """Return live local L1 and per-adapter L2 capacity pressure.

    Collection is delegated to the storage manager and offloaded because
    backend status calls are synchronous. A failure in one tier is represented
    as an ``unknown`` entry; only failure to produce the top-level snapshot
    makes the endpoint unavailable.

    Args:
        request: FastAPI request carrying the typed MP HTTP context.

    Returns:
        The node-local memory-pressure report.

    Raises:
        HTTPException: 503 before initialization or when the storage manager
            cannot produce a snapshot.
    """
    context = get_context(request)
    if context.memory_pressure is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="memory pressure snapshot unavailable",
        )
    try:
        return await asyncio.to_thread(context.memory_pressure.snapshot)
    except MemoryPressureUnavailable as exc:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="memory pressure snapshot unavailable",
        ) from exc
