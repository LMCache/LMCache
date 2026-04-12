# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import Optional
import argparse
import asyncio
import hashlib
import sys

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    compress_slot_mapping,
    parse_mixed_slot_mapping,
)
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


@app.get("/api/kvcache/check")
async def kvcache_check(
    request: Request,
    slot_mapping: Optional[str] = None,
    chunk_size: Optional[int] = None,
    instance_id: int = 0,
    layerwise: bool = False,
):
    """Compute MD5 checksums for KV cache at specified slots.

    Uses the same slot_mapping format as the vLLM cache API:
    comma-separated integers and [start,end] range expressions.

    Args:
        slot_mapping: Slot indices in mixed format.
            Examples: "0,1,2,3", "[0,511]",
            "1,2,[9,12],17".
        chunk_size: Group slots into chunks of this size.
        instance_id: GPU context instance ID (default 0).
        layerwise: Per-layer checksums if True.

    Example::

        curl "http://localhost:8080/api/kvcache/check?\
    slot_mapping=[0,511]&chunk_size=256"
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )

    ctx = engine.gpu_contexts.get(instance_id)
    if ctx is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": ("instance_id %d not registered" % instance_id),
            },
        )

    if not slot_mapping:
        return JSONResponse(
            status_code=400,
            content={"error": "slot_mapping is required"},
        )

    slot_indices, error_info = parse_mixed_slot_mapping(
        slot_mapping,
    )
    if error_info:
        return JSONResponse(
            status_code=400,
            content=error_info,
        )
    assert slot_indices is not None

    if chunk_size is None or chunk_size <= 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "chunk_size must be positive",
            },
        )

    kv_tensors = ctx.kv_tensors
    if not kv_tensors:
        return JSONResponse(
            status_code=404,
            content={"error": "kv_caches empty"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_mp_checksums(
            kv_tensors,
            slot_indices,
            chunk_size,
            layerwise,
        ),
    )

    # Include compressed slot_mapping_ranges in response
    result["slot_mapping_ranges"] = compress_slot_mapping(
        slot_indices,
    )

    return JSONResponse(content=result)


def _compute_mp_checksums(
    kv_tensors: list,
    slot_indices: list[int],
    chunk_size: int,
    layerwise: bool,
) -> dict:
    """Compute MD5 checksums over KV cache slots.

    Each kv_tensor has shape [2, NB, BS, NH, HS]
    (NL_X_TWO_NB_BS_NH_HS format).  A *slot* index maps
    to (block_id, block_offset) via divmod by BS.
    """
    num_slots = len(slot_indices)
    num_chunks = (num_slots + chunk_size - 1) // chunk_size
    slot_t = torch.tensor(slot_indices, dtype=torch.long)

    # Extract per-layer data at the requested slots
    # kv: [2, NB, BS, NH, HS] -> [2, NB*BS, NH, HS]
    layer_data: list[torch.Tensor] = []
    for kv in kv_tensors:
        reshaped = kv.reshape(2, -1, *kv.shape[3:])
        layer_data.append(reshaped[:, slot_t, ...])

    if layerwise:
        checksums: dict[str, list[str]] = {}
        for li, ld in enumerate(layer_data):
            name = "layer_%d" % li
            cks: list[str] = []
            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, num_slots)
                chunk = ld[:, s:e, ...].contiguous()
                cks.append(hashlib.md5(chunk.numpy().tobytes()).hexdigest())
            checksums[name] = cks
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": checksums,
            "layerwise": True,
        }
    else:
        cks_list: list[str] = []
        for ci in range(num_chunks):
            s = ci * chunk_size
            e = min(s + chunk_size, num_slots)
            md5 = hashlib.md5()
            for ld in layer_data:
                chunk = ld[:, s:e, ...].contiguous()
                md5.update(chunk.numpy().tobytes())
            cks_list.append(md5.hexdigest())
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": cks_list,
            "layerwise": False,
        }


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
    # Use asyncio.run() directly instead of server.run()
    # to avoid compatibility issues with PyCharm's
    # debugger patching asyncio.
    # Ensure uvloop (if installed) is still used as the
    # event loop implementation.
    if hasattr(config, "get_loop_factory"):
        # uvicorn >= 0.36.0
        loop_factory = config.get_loop_factory()
        if sys.version_info >= (3, 12):
            asyncio.run(server.serve(), loop_factory=loop_factory)
        else:
            # loop_factory kwarg requires Python 3.12+;
            # fall back to setting the event loop manually.
            loop = loop_factory()
            loop.run_until_complete(server.serve())
            loop.close()
    else:
        # uvicorn < 0.36.0
        config.setup_event_loop()
        asyncio.run(server.serve())


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
