# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
import argparse
import json

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
from lmcache.v1.utils.json_utils import make_json_safe, safe_asdict

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


def _serialize_configs(configs: dict[str, Any]) -> dict[str, Any]:
    """Convert a mapping of config dataclasses into a JSON-safe dict.

    Args:
        configs: Mapping from config name to dataclass instance (or
            already JSON-safe value).

    Returns:
        A dict where every value is composed solely of JSON-native types,
        suitable for ``json.dumps``.
    """
    payload: dict[str, Any] = {}
    for name, cfg in configs.items():
        if is_dataclass(cfg) and not isinstance(cfg, type):
            payload[name] = safe_asdict(cfg)
        else:
            payload[name] = make_json_safe(cfg)
    return payload


def _resolve_config_dump_path(dump_path: str, http_port: int) -> Path:
    """Resolve the path used to persist the server configuration.

    Args:
        dump_path: User-supplied path. Empty string selects the default
            location, namespaced by ``http_port`` so multiple servers on
            the same host do not overwrite each other.
        http_port: HTTP port the server is binding to.

    Returns:
        The absolute filesystem path to write the JSON config to.
    """
    if dump_path == "":
        return Path(f"/tmp/lmcache-config-{http_port}.json")
    return Path(dump_path)


def _write_config_dump(configs: dict[str, Any], path: Path) -> None:
    """Write the serialized server configuration to ``path``.

    Args:
        configs: Mapping of config name to dataclass instance.
        path: Filesystem location for the JSON dump.

    Raises:
        OSError: If the parent directory cannot be created or the file
            cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize_configs(configs), indent=2, sort_keys=True))


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

    dump_path = _resolve_config_dump_path(
        http_config.config_dump_path, http_config.http_port
    )
    try:
        _write_config_dump(_configs, dump_path)
        logger.info("Wrote LMCache server config dump to %s", dump_path)
    except OSError as exc:
        logger.warning(
            "Failed to write LMCache server config dump to %s: %s",
            dump_path,
            exc,
        )

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
