# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
import argparse

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_PROMETHEUS_CONFIG
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
)
from lmcache.v1.multiprocess.server import run_cache_server

logger = init_logger(__name__)


# ----------------------------
# Server configuration
# ----------------------------
@dataclass
class ServerConfig:
    """Configuration for the HTTP server and ZMQ backend."""

    zmq_host: str = "localhost"
    zmq_port: int = 5555
    chunk_size: int = 256
    cpu_buffer_size: float = 5.0
    max_workers: int = 1

    def to_storage_manager_config(self) -> StorageManagerConfig:
        return StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=int(self.cpu_buffer_size * 1024**3),
                    use_lazy=True,
                ),
            ),
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
            ),
        )


_server_config = ServerConfig()


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
    logger.info("Starting LMCache HTTP server...")
    zmq_server, engine = run_cache_server(
        storage_manager_config=_server_config.to_storage_manager_config(),
        prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
        host=_server_config.zmq_host,
        port=_server_config.zmq_port,
        chunk_size=_server_config.chunk_size,
        max_workers=_server_config.max_workers,
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


def run_http_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    zmq_host: str = "localhost",
    zmq_port: int = 5555,
    chunk_size: int = 256,
    cpu_buffer_size: float = 5.0,
    max_workers: int = 1,
):
    """
    Run the LMCache HTTP server with integrated MP (ZMQ) server.

    Args:
        host: HTTP server host
        port: HTTP server port
        zmq_host: ZMQ server host
        zmq_port: ZMQ server port
        chunk_size: Chunk size for KV cache operations
        cpu_buffer_size: CPU buffer size in GB
        max_workers: Maximum number of worker threads for ZMQ server
    """
    global _server_config

    _server_config.zmq_host = zmq_host
    _server_config.zmq_port = zmq_port
    _server_config.chunk_size = chunk_size
    _server_config.cpu_buffer_size = cpu_buffer_size
    _server_config.max_workers = max_workers

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    logger.info("Starting LMCache HTTP server on http://%s:%d", host, port)
    server.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache HTTP Server with integrated MP Cache Server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the HTTP server"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the HTTP server"
    )
    parser.add_argument(
        "--zmq-host", type=str, default="localhost", help="Host to bind the ZMQ server"
    )
    parser.add_argument(
        "--zmq-port", type=int, default=5555, help="Port to bind the ZMQ server"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256, help="Chunk size for KV cache operations"
    )
    parser.add_argument(
        "--cpu-buffer-size", type=float, default=5.0, help="CPU buffer size in GB"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum number of worker threads"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_http_server(
        host=args.host,
        port=args.port,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        max_workers=args.max_workers,
    )
