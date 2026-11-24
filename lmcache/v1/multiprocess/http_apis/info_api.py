# SPDX-License-Identifier: Apache-2.0
"""Basic-info endpoints: liveness, health, status, and version.

These are the small, dependency-light routes used to inspect that the
server is up and what it is running:

- ``GET /`` — static liveness payload (does not touch the engine).
- ``GET /livez`` — lightweight liveness probe; responds if the HTTP
  server and cache engine are alive. No backend storage, no worker
  inventory, no I/O of any kind — just the thin check that the engine
  object is wired into the FastAPI app state. Kubernetes uses this to
  decide whether to restart the pod.
- ``GET /readyz`` — comprehensive readiness probe. Aggregates the
  component tree from :meth:`MPCacheServer.report_status` (storage
  manager health, L1 managers, L2 adapters, controllers) and reports
  the worker liveness summary. Skips expensive operations such as L1
  ``memcheck()`` so the probe stays fast under load. Kubernetes uses
  this to decide whether to route traffic to the pod.
- ``GET /healthcheck`` — legacy liveness/readiness probe preserved
  for backwards compatibility. Returns ``200`` when the engine exists,
  ``503`` before startup completes. New deployments should migrate to
  ``/livez`` and ``/readyz``.
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
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()

# Thresholds for /readyz worker inventory check. When the MP server
# has liveness targets configured, we require at least one tracked
# worker to consider ourselves ready to serve requests. A count of
# zero means either the server is still starting up or all workers
# have been reaped — either way we should not advertise readiness.
# The threshold itself is intentionally a simple integer comparison
# rather than a configurable knob; it mirrors what the internal
# ``worker_liveness`` summary already reports.
_MIN_TRACKED_INSTANCES_FOR_READY = 1


@router.get("/")
async def root() -> dict[str, str]:
    """
    Basic liveness check endpoint.
    Returns:
        dict: A dictionary containing the status and service name.
    """
    return {"status": "ok", "service": "LMCache HTTP API"}


@router.get("/livez")
async def livez(request: Request) -> Any:
    """Lightweight liveness probe for Kubernetes.

    Returns ``200`` when the cache engine is wired into the FastAPI app
    state. Does not inspect storage, workers, or any backend component —
    the goal is to detect a process that is completely wedged while
    keeping the check cheap enough for a high-frequency probe.

    Args:
        request: The incoming HTTP request; ``app.state.engine`` carries
            the :class:`MPCacheServer` when the server is running.

    Returns:
        ``{"status": "alive"}`` with HTTP 200 when the engine exists.
        ``{"status": "not_alive", "reason": "..."}`` with HTTP 503 before
        the engine is initialized or after a critical failure.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_alive",
                "reason": "engine not initialized",
            },
        )

    return {"status": "alive"}


@router.get("/readyz")
async def readyz(request: Request) -> Any:
    """Comprehensive readiness probe for Kubernetes.

    Aggregates every component the server needs to serve requests:

    - The :class:`MPCacheServer` must be wired into the app state.
    - The storage manager's ``is_healthy`` flag (computed across L1
      managers, store controllers, eviction controllers, prefetch,
      L2 eviction, and all L2 adapters) must be ``True``.
    - When liveness targets are configured, at least one tracked
      worker must be alive. A count of zero signals either a
      still-starting server or a fully reaped worker pool — both
      mean we are not ready to serve.

    The probe deliberately skips expensive operations (e.g. L1
    ``memcheck()``) and only reads state already held in memory.

    Args:
        request: The incoming HTTP request; ``app.state.engine`` carries
            the :class:`MPCacheServer` whose status we aggregate.

    Returns:
        HTTP 200 with ``{"status": "ready", "checks": {...}}`` when every
        component is healthy. HTTP 503 with
        ``{"status": "not_ready", "checks": {...}, "failed": [...]}``
        listing every failing check.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {"engine": "missing"},
                "failed": ["engine"],
            },
        )

    try:
        full_status = engine.report_status()
    except Exception:
        logger.exception("readyz: engine.report_status() raised")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {"report_status": "raised exception"},
                "failed": ["report_status"],
            },
        )

    checks: dict[str, Any] = {}
    failed: list[str] = []

    # Component tree health: the storage manager aggregates health flags
    # across every L1 manager, store controller, eviction controller,
    # prefetch controller, L2 eviction controller, and L2 adapter.
    storage_healthy = bool(full_status.get("is_healthy", False))
    checks["storage_manager"] = storage_healthy
    if not storage_healthy:
        failed.append("storage_manager")

    # Worker liveness: when the MP server tracks worker heartbeats
    # (reaper enabled), require at least one tracked instance. The
    # liveness_targets list is empty on servers that never registered
    # workers (e.g. standalone local mode), so we only enforce this
    # when the summary is actually present.
    worker_liveness = full_status.get("worker_liveness")
    if worker_liveness is not None:
        tracked = worker_liveness.get("tracked_instances", 0)
        checks["workers"] = {
            "enabled": worker_liveness.get("enabled", False),
            "tracked_instances": tracked,
            "reap_timeout_seconds": worker_liveness.get("reap_timeout_seconds", 0),
        }
        if tracked < _MIN_TRACKED_INSTANCES_FOR_READY:
            failed.append("workers")

    status_code = 200 if not failed else 503
    response_body: dict[str, Any] = {
        "status": "ready" if not failed else "not_ready",
        "checks": checks,
    }
    if failed:
        response_body["failed"] = failed

    return JSONResponse(status_code=status_code, content=response_body)


@router.get("/healthcheck")
async def healthcheck(request: Request) -> Any:
    """Legacy health check endpoint preserved for backwards compatibility.

    Checks:
        - HTTP server is alive (implicit: if you get a response)
        - Cache engine is alive

    New deployments should use ``GET /livez`` for liveness and
    ``GET /readyz`` for readiness instead.
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
