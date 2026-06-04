# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
import argparse

# Third Party
from fastapi import FastAPI
import uvicorn

# First Party
from lmcache import torch_dev
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
from lmcache.v1.multiprocess.http_api_registry import (
    HTTPAPIRegistry,
)
from lmcache.v1.multiprocess.mp_runtime_plugin_launcher import (
    MPRuntimePluginLauncher,
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
        "Starting LMCache HTTP server... (accelerator available: %s)",
        torch_dev.is_available(),
    )
    mp_config = _configs["mp"]

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
