# SPDX-License-Identifier: Apache-2.0
"""Basic-info endpoints: liveness, health, status, and version.

These are the small, dependency-light routes used to inspect that the
server is up and what it is running:

- ``GET /`` — static liveness payload (does not touch the engine).
- ``GET /healthcheck`` — Kubernetes liveness probe.
- ``GET /readyz`` — Kubernetes readiness probe, per-controller liveness.
- ``GET /status`` — detailed internal state of all MP components.
- ``GET /version``, ``/lmc_version``, ``/commit_id`` — version descriptors,
  re-exposed from the shared ``internal_api_server.vllm`` version router.
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.internal_api_server.vllm.version_api import router as _version_router

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
    Health check endpoint for k8s liveness probes.

    Checks:
        - HTTP server is alive (implicit: if you get a response)
        - Cache engine is alive

    This stays deliberately shallow so a transient storage fault does not
    restart the container. Use ``/readyz`` to gate traffic on component
    health.
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


@router.get("/readyz")
async def readyz(request: Request) -> Any:
    """
    Readiness probe reporting which storage controllers are running.

    Unlike ``/healthcheck``, which only confirms the engine was built, this
    checks the background threads the data path depends on: the store,
    prefetch and eviction controllers. Allocator consistency and L2 backend
    health are deliberately left to ``/status``; both cost too much to answer
    on a probe interval, and a failing external backend should not drain
    traffic from a server whose L1 still serves.

    Returns:
        ``200`` with ``status: ready`` when every controller is running, or
        ``503`` with ``status: not_ready`` and the stopped controller names
        in ``unhealthy``. ``components`` always maps every controller name to
        its liveness flag.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "engine not initialized",
            },
        )

    components = engine.report_readiness()
    unhealthy = sorted(name for name, healthy in components.items() if not healthy)
    if unhealthy:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "components": components,
                "unhealthy": unhealthy,
            },
        )
    return {"status": "ready", "components": components}


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
