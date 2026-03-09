# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
import argparse

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
    PrometheusConfig,
    add_prometheus_args,
    parse_args_to_prometheus_config,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
)
from lmcache.v1.multiprocess.config import (
    HTTPFrontendConfig,
    MPServerConfig,
    add_http_frontend_args,
    add_mp_server_args,
    parse_args_to_http_frontend_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.server import run_cache_server

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
    zmq_server, engine = run_cache_server(
        mp_config=_configs["mp"],
        storage_manager_config=_configs["storage_manager"],
        prometheus_config=_configs["prometheus"],
        return_engine=True,
    )
    app.state.zmq_server = zmq_server
    app.state.engine = engine
    logger.info("LMCache HTTP server initialized")

    yield

    # Shutdown
    logger.info("Shutting down LMCache HTTP server...")
    get_prometheus_controller().stop()
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

    if not engine.storage_manager.memcheck():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "memory check failed"},
        )

    return {"status": "healthy"}


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


def run_http_server(
    http_config: HTTPFrontendConfig,
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    prometheus_config: PrometheusConfig,
) -> None:
    """
    Run the LMCache HTTP server with integrated MP (ZMQ) server.

    Args:
        http_config: Configuration for the HTTP frontend
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        prometheus_config: Configuration for the Prometheus observability stack
    """
    _configs["mp"] = mp_config
    _configs["storage_manager"] = storage_manager_config
    _configs["prometheus"] = prometheus_config

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
    add_prometheus_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    http_config = parse_args_to_http_frontend_config(args)
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    prometheus_config = parse_args_to_prometheus_config(args)
    run_http_server(
        http_config=http_config,
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        prometheus_config=prometheus_config,
    )
