# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import Optional
import argparse
import asyncio
import hashlib

# Third Party
from fastapi import FastAPI
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
from lmcache.v1.multiprocess.http_api_registry import (
    HTTPAPIRegistry,
)
from lmcache.v1.multiprocess.mp_runtime_plugin_launcher import (
    MPRuntimePluginLauncher,
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

    result = run_cache_server(
        mp_config=mp_config,
        storage_manager_config=_configs["storage_manager"],
        obs_config=_configs["observability"],
        return_engine=True,
        start_prometheus_http_server=False,
    )
    assert result is not None, "run_cache_server returned None with return_engine=True"
    zmq_server, engine = result

    # Launch runtime plugins if configured. Plugins receive the full
    # server config (including HTTP host/port) via the
    # LMCACHE_RUNTIME_PLUGIN_CONFIG environment variable.
    plugin_launcher = None
    if mp_config.runtime_plugin_config.locations:
        extra_kwargs = {}
        http_config = _configs.get("http")
        if http_config is not None:
            extra_kwargs["http_config"] = http_config
        plugin_launcher = MPRuntimePluginLauncher(
            runtime_plugin_config=mp_config.runtime_plugin_config,
            mp_config=mp_config,
            storage_manager_config=_configs["storage_manager"],
            obs_config=_configs["observability"],
            **extra_kwargs,
        )
        plugin_launcher.launch_plugins()

    app.state.zmq_server = zmq_server
    app.state.engine = engine
    app.state.plugin_launcher = plugin_launcher
    logger.info("LMCache HTTP server initialized")

    yield

    # Shutdown
    logger.info("Shutting down LMCache HTTP server...")
    launcher = getattr(app.state, "plugin_launcher", None)
    if launcher is not None:
        launcher.stop_plugins()
    get_event_bus().stop()
    if hasattr(app.state, "zmq_server") and app.state.zmq_server is not None:
        app.state.zmq_server.close()
    logger.info("LMCache HTTP server stopped")


app = FastAPI(title="LMCache HTTP API", version="1.0.0", lifespan=lifespan)

# Automatically discover and register all HTTP API endpoints
registry = HTTPAPIRegistry(app)
registry.register_all_apis()


@app.get("/api/kvcache/check")
async def kvcache_check(
    request: Request,
    slot_mapping: Optional[str] = None,
    chunk_size: Optional[int] = None,
    instance_id: int = 0,
    layerwise: bool = False,
) -> JSONResponse:
    """Compute MD5 checksums for KV cache slots.

    Args:
        slot_mapping: Slot indices (mixed format).
        chunk_size: Group slots into chunks.
        instance_id: GPU context ID (default 0).
        layerwise: Per-layer checksums if True.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )

    gpu_ctxs = getattr(engine, "gpu_contexts", None)
    if gpu_ctxs is None:
        return JSONResponse(
            status_code=501,
            content={
                "error": "checksum not supported for this engine type",
            },
        )

    ctx = gpu_ctxs.get(instance_id)
    if ctx is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "instance_id %d not registered" % instance_id,
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
        logger.warning(
            "Invalid slot_mapping from client: %s",
            error_info,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid slot_mapping format",
            },
        )
    if slot_indices is None:
        return JSONResponse(
            status_code=400,
            content={"error": "failed to parse slot_mapping"},
        )

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

    Each kv_tensor shape: [2, NB, BS, NH, HS].
    Slots are mapped via reshape to [2, NB*BS, NH, HS].
    """
    num_slots = len(slot_indices)
    num_chunks = (num_slots + chunk_size - 1) // chunk_size
    slot_t = torch.tensor(
        slot_indices,
        dtype=torch.long,
    )

    # kv: [2, NB, BS, NH, HS] -> [2, NB*BS, NH, HS]
    layer_data: list[torch.Tensor] = []
    for kv in kv_tensors:
        flat = kv.reshape(2, -1, *kv.shape[3:])
        # Move to CPU once per layer to save GPU memory
        # and avoid repeated transfers in the chunking loop
        sliced = flat[:, slot_t, ...].cpu()
        # Handle BFloat16 which is not supported by numpy
        if sliced.dtype == torch.bfloat16:
            sliced = sliced.to(torch.float32)
        layer_data.append(sliced)

    if layerwise:
        checksums: dict[str, list[str]] = {}
        for li, ld in enumerate(layer_data):
            cks: list[str] = []
            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, num_slots)
                chunk = ld[:, s:e, ...].contiguous()
                cks.append(hashlib.md5(chunk.numpy().tobytes()).hexdigest())
            checksums["layer_%d" % li] = cks
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": checksums,
            "layerwise": True,
        }

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
    _configs["http"] = http_config
    app.state.configs = _configs

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
