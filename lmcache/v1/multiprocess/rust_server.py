# SPDX-License-Identifier: Apache-2.0
"""Configuration and engine construction for the Rust-owned server lifecycle."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

# Third Party
import uvicorn
from fastapi import FastAPI

# First Party
from lmcache.v1.distributed.config import StorageManagerConfig, parse_args_to_config
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.mp_observability.gc_monitor import shutdown_gc_monitor
from lmcache.v1.mp_observability.otel_init import shutdown_grpc_otel_metrics
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    parse_args_to_coordinator_config,
    parse_args_to_http_frontend_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.http_auth import HTTPAuthenticationConfig
from lmcache.v1.multiprocess.http_server import create_http_server, parse_args
from lmcache.v1.multiprocess.server import MPCacheServer, run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestServer


def shutdown_observability() -> None:
    """Stop process-wide observability workers after Rust closes the engine.

    Called after the HTTP worker, event bus and MP resources have stopped.
    Both observability subsystems tolerate calls when startup failed.
    """
    try:
        shutdown_gc_monitor()
    finally:
        shutdown_grpc_otel_metrics()


@dataclass
class ServerBootstrap:
    """Validated configuration and an unstarted HTTP server owned by Rust."""

    http_server: uvicorn.Server
    mp_config: MPServerConfig
    storage_manager_config: StorageManagerConfig
    obs_config: ObservabilityConfig
    coordinator_config: CoordinatorConfig

    def start_engine(self) -> tuple[RequestServer, MPCacheServer, EventBus]:
        """Start MP resources and lend them to the HTTP application.

        Returns:
            Request server, engine and event bus. Rust owns them and must stop
            the event bus, close the request server, close the engine, and call
            ``shutdown_observability`` after joining the HTTP worker.

        Raises:
            Exception: Existing MP engine or transport initialization failed.
                The Rust caller must still call ``shutdown_observability``.
        """
        result = run_cache_server(
            mp_config=self.mp_config,
            storage_manager_config=self.storage_manager_config,
            obs_config=self.obs_config,
            return_engine=True,
            start_prometheus_http_server=False,
            coordinator_config=self.coordinator_config,
        )
        assert result is not None, "return_engine=True must return MP resources"
        cast(
            FastAPI, self.http_server.config.app
        ).state.externally_managed_cache_server = result
        return *result, get_event_bus()


def prepare_server(arguments: Sequence[str]) -> ServerBootstrap:
    """Parse existing server flags and construct a validated bootstrap.

    Args:
        arguments: CLI tokens after the Rust executable name.

    Returns:
        Configuration and an HTTP server; no engine or listener is started.

    Raises:
        SystemExit: Help was requested or CLI arguments are invalid.
        ValueError: Server configuration or authentication is invalid.
    """
    args = parse_args(arguments)
    mp_config = parse_args_to_mp_server_config(args)
    storage_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    coordinator_config = parse_args_to_coordinator_config(args)
    http_server = create_http_server(
        http_config=parse_args_to_http_frontend_config(args),
        mp_config=mp_config,
        storage_manager_config=storage_config,
        obs_config=obs_config,
        coordinator_config=coordinator_config,
        http_authentication=HTTPAuthenticationConfig.from_sources(args.api_key_file),
    )
    # create_http_server may replace the storage policies for P2P writes.
    configs = cast(FastAPI, http_server.config.app).state.configs
    return ServerBootstrap(
        http_server,
        mp_config,
        configs["storage_manager"],
        obs_config,
        coordinator_config,
    )
