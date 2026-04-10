# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
import argparse

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.multiprocess.config import (
    HTTPFrontendConfig,
    MPServerConfig,
    add_http_frontend_args,
    add_mp_server_args,
    parse_args_to_http_frontend_config,
    parse_args_to_mp_server_config,
)

logger = init_logger(__name__)


# Module-level config holders, set by run_http_server() before FastAPI startup.
# Stored in a dict so the lifespan closure captures the mutable container.
_configs: dict = {}


# ----------------------------
# FastAPI lifespan for initialization and cleanup
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the LMCache HTTP server.

    On startup: Initialize ZMQ server and cache engine.
    On shutdown: Clean up ZMQ server resources.
    """
    # Startup
    logger.info(
        "Starting LMCache HTTP server... (CUDA available: %s)",
        torch.cuda.is_available(),
    )
    mp_config = _configs["mp"]
    if mp_config.engine_type == "blend":
        # First Party
        from lmcache.v1.multiprocess.blend_server_v2 import run_cache_server
    else:
        # First Party
        from lmcache.v1.multiprocess.server import run_cache_server

    zmq_server, engine = run_cache_server(
        mp_config=mp_config,
        storage_manager_config=_configs["storage_manager"],
        obs_config=_configs["observability"],
        return_engine=True,
    )
    app.state.zmq_server = zmq_server
    app.state.engine = engine
    logger.info("LMCache HTTP server initialized")

    yield

    # Shutdown
    logger.info("Shutting down LMCache HTTP server...")
    get_event_bus().stop()
    if hasattr(app.state, "zmq_server") and app.state.zmq_server is not None:
        app.state.zmq_server.close()
    logger.info("LMCache HTTP server stopped")


app = FastAPI(title="LMCache HTTP API", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "service": "LMCache HTTP API"}


@app.get("/api/healthcheck")
async def healthcheck(request: Request):
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
            content={"status": "unhealthy", "reason": "engine not initialized"},
        )

    return {"status": "healthy"}


@app.post("/api/clear-cache")
async def clear_cache(request: Request):
    """
    Force-clear all KV cache data stored in L1 (CPU) memory.

    This clears all objects including those with active read/write locks.
    In-flight store or prefetch operations may be corrupted.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "engine not initialized"},
        )

    engine.clear()
    logger.info("Cache cleared via HTTP API")
    return {"status": "ok"}


@app.get("/api/status")
async def status(request: Request):
    """
    Detailed status endpoint for inspecting internal state of all
    MP components (L1 cache, L2 adapters, controllers, sessions).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.report_status()


# ----------------------------
# Per-user quota management endpoints
# ----------------------------


class QuotaSetRequest(BaseModel):
    """Request body for setting a user's quota."""

    limit_gb: float


_DEFAULT_USER_SENTINEL = "_default"
"""URL path sentinel that maps to user_id="".

Empty strings cannot be used as URL path parameters, so the API uses
``_default`` in the URL to represent the empty-user-id namespace
(legacy/anonymous traffic).
"""


def _resolve_user_id(user_id: str) -> str:
    """Map the URL path sentinel to the actual user_id.

    Args:
        user_id: The user_id from the URL path.

    Returns:
        The resolved user_id (empty string if sentinel was used).
    """
    return "" if user_id == _DEFAULT_USER_SENTINEL else user_id


def _get_quota_manager(request: Request):
    """Extract the QuotaManager from the engine, or None.

    Args:
        request: The incoming HTTP request.

    Returns:
        QuotaManager instance or None.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return None
    return engine.get_quota_manager()


def _get_per_user_usage(request: Request) -> dict[str, tuple[float, float]]:
    """Aggregate per-user usage across all L2 adapters.

    Args:
        request: The incoming HTTP request.

    Returns:
        Mapping of user_id to (current_bytes, bytes_after_eviction).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {}
    combined: dict[str, tuple[float, float]] = {}
    for adapter in engine.get_l2_adapters():
        for uid, (cur, after) in adapter.get_per_user_usage().items():
            prev_cur, prev_after = combined.get(uid, (0.0, 0.0))
            combined[uid] = (prev_cur + cur, prev_after + after)
    return combined


@app.put("/api/quota/{user_id}")
async def set_quota(user_id: str, body: QuotaSetRequest, request: Request):
    """Set or update the storage quota for a user.

    Use ``_default`` as user_id to set quota for the empty-user-id
    namespace (legacy/anonymous traffic).

    Args:
        user_id: The user to set a quota for.
        body: JSON body with ``limit_gb`` field.
        request: The incoming HTTP request.

    Returns:
        JSON with user_id, limit_gb, and status.
    """
    user_id = _resolve_user_id(user_id)
    qm = _get_quota_manager(request)
    if qm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "quota manager not available"},
        )
    try:
        qm.set_quota(user_id, body.limit_gb)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "reason": str(e)},
        )
    return {"user_id": user_id, "limit_gb": body.limit_gb, "status": "ok"}


@app.get("/api/quota/{user_id}")
async def get_quota(user_id: str, request: Request):
    """Get quota and current usage for a user.

    Use ``_default`` as user_id to query the empty-user-id namespace.

    Args:
        user_id: The user to query.
        request: The incoming HTTP request.

    Returns:
        JSON with user_id, limit_gb, current_usage_gb, and exists flag.
    """
    user_id = _resolve_user_id(user_id)
    qm = _get_quota_manager(request)
    if qm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "quota manager not available"},
        )
    limit_bytes = qm.get_limit_bytes(user_id)
    exists = limit_bytes > 0

    per_user_usage = _get_per_user_usage(request)
    current_bytes, _ = per_user_usage.get(user_id, (0.0, 0.0))

    return {
        "user_id": user_id,
        "limit_gb": limit_bytes / (1024**3),
        "current_usage_gb": current_bytes / (1024**3),
        "exists": exists,
    }


@app.delete("/api/quota/{user_id}")
async def delete_quota(user_id: str, request: Request):
    """Remove the quota entry for a user.

    The user's cached data will be evicted at the next eviction cycle.
    Use ``_default`` as user_id for the empty-user-id namespace.

    Args:
        user_id: The user whose quota should be removed.
        request: The incoming HTTP request.

    Returns:
        JSON with user_id and status.
    """
    user_id = _resolve_user_id(user_id)
    qm = _get_quota_manager(request)
    if qm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "quota manager not available"},
        )
    qm.remove_quota(user_id)
    return {"user_id": user_id, "status": "removed"}


@app.get("/api/quota")
async def list_quotas(request: Request):
    """List all registered quotas with per-user usage.

    Args:
        request: The incoming HTTP request.

    Returns:
        JSON with a ``users`` mapping of user_id to limit_gb and
        current_usage_gb.
    """
    qm = _get_quota_manager(request)
    if qm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "quota manager not available"},
        )
    all_quotas = qm.get_all_quotas()
    per_user_usage = _get_per_user_usage(request)

    users = {}
    for uid, limit_bytes in all_quotas.items():
        current_bytes, _ = per_user_usage.get(uid, (0.0, 0.0))
        users[uid] = {
            "limit_gb": limit_bytes / (1024**3),
            "current_usage_gb": current_bytes / (1024**3),
        }
    return {"users": users}


def run_http_server(
    http_config: HTTPFrontendConfig,
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
) -> None:
    """
    Run the LMCache HTTP server with integrated MP (ZMQ) server.

    Args:
        http_config: Configuration for the HTTP frontend
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
    """
    _configs["mp"] = mp_config
    _configs["storage_manager"] = storage_manager_config
    _configs["observability"] = obs_config

    config = uvicorn.Config(
        app=app,
        host=http_config.http_host,
        port=http_config.http_port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    logger.info(
        "Starting LMCache HTTP server on http://%s:%d",
        http_config.http_host,
        http_config.http_port,
    )
    server.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache HTTP Server with integrated MP Cache Server"
    )
    add_http_frontend_args(parser)
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    http_config = parse_args_to_http_frontend_config(args)
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_http_server(
        http_config=http_config,
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
