# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory for the mp coordinator.

The coordinator is a FastAPI app. Endpoints are auto-discovered from the
``http_apis`` package (the same convention as the mp server's HTTP API) and stay
thin, operating on the shared collaborators carried on ``app.state``: ``config``,
``registry``, ``key_directory``, ``eviction_controller`` (which owns quota and
usage), and the ingest layer's ``event_gate``.
The lifespan runs background tasks for health-checking (eviction of instances
whose heartbeats have lapsed) and the fleet L2 eviction control loop, which the
controller owns (``FleetEvictionController.run``).

Adding a capability = a new ``http_apis/<name>_api.py`` router (auto-discovered)
that uses those shared collaborators. To push to an mp server, a future router
resolves the instance's address from the registry (``ip`` + ``http_port``) and
POSTs to that server's specific endpoint. A domain with real logic/state of its
own adds a module under ``controllers/`` stashed on ``app.state`` here; thin
domains (like membership) just use the registry directly.
"""

# Standard
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import contextlib

# Third Party
from fastapi import FastAPI
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.utils.router_discovery import discover_api_routers

logger = init_logger(__name__)


def evict_stale(registry: InstanceRegistry, instance_timeout: float) -> list[str]:
    """Deregister every instance whose heartbeat is older than the timeout.

    Args:
        registry: The shared instance registry.
        instance_timeout: Max seconds since the last heartbeat before eviction.

    Returns:
        The ids of instances evicted in this sweep.
    """
    evicted = []
    for instance_id in registry.stale(instance_timeout):
        if registry.deregister(instance_id) is not None:
            logger.warning("Instance %s timed out; evicted", instance_id)
            evicted.append(instance_id)
    return evicted


def create_app(config: MPCoordinatorConfig) -> FastAPI:
    """Build the coordinator FastAPI app.

    Args:
        config: The coordinator configuration.

    Returns:
        A configured FastAPI application. ``app.state`` carries the shared
        collaborators (``config`` plus the :class:`CoordinatorContext`); all
        ``http_apis`` routers are registered.
    """
    registry = InstanceRegistry()
    key_directory = KeyDirectory()
    if config.enable_blend_lookup:
        # Only now does the directory hash chunk content: chunk_size is the
        # match window (the fleet chunk), blend_probe_stride the probe density.
        key_directory.enable_blend_lookup(
            chunk_size=config.chunk_size, probe_stride=config.blend_probe_stride
        )
    eviction_controller = FleetEvictionController(
        eviction_ratio=config.eviction_ratio,
        trigger_watermark=config.trigger_watermark,
    )
    prefetch_manager = PrefetchManager()
    # Resolves pin requests' token_ids to object keys; must match the fleet's
    # chunk size and hash algorithm (see MPCoordinatorConfig).
    token_hasher = TokenHasher(
        chunk_size=config.chunk_size, hash_algorithm=config.hash_algorithm
    )
    # Ingest layer: the gate admits, the broadcaster fans out. Adding a
    # consumer of the fleet's cache-event stream is a register call here.
    event_broadcaster = CacheEventBroadcaster()
    event_broadcaster.register_consumer(key_directory)
    event_broadcaster.register_consumer(eviction_controller)
    event_gate = EventGate(event_broadcaster)

    ctx = CoordinatorContext(
        registry=registry,
        eviction_controller=eviction_controller,
        prefetch_manager=prefetch_manager,
        token_hasher=token_hasher,
        key_directory=key_directory,
        event_gate=event_gate,
    )

    async def _health_loop() -> None:
        """Evict stale instances on a timer until cancelled."""
        while True:
            await asyncio.sleep(config.health_check_interval)
            evict_stale(registry, config.instance_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start background tasks and clean up resources on shutdown."""
        # Shared async client for outbound coordinator → MP server
        # calls (eviction dispatch). Created inside the lifespan so it
        # binds to the running event loop.
        outbound_client = httpx.AsyncClient(timeout=30.0)
        app.state.outbound_client = outbound_client
        health_task = None
        eviction_task = None
        if config.health_check_interval > 0:
            health_task = asyncio.create_task(_health_loop())
        if config.eviction_check_interval > 0:
            eviction_task = asyncio.create_task(
                eviction_controller.run(
                    registry, outbound_client, config.eviction_check_interval
                )
            )
        logger.info(
            "MP coordinator listening on http://%s:%d", config.host, config.port
        )
        try:
            yield
        finally:
            for task in (health_task, eviction_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            await eviction_controller.wait_for_in_flight_dispatches()
            await outbound_client.aclose()

    app = FastAPI(title="LMCache MP Coordinator", version="1.0.0", lifespan=lifespan)
    app.state.ctx = ctx
    # Out-of-context collaborator kept on app.state directly.
    app.state.config = config

    apis_path = Path(__file__).parent / "http_apis"
    package = f"{__package__}.http_apis"
    for router in discover_api_routers(apis_path, package):
        app.include_router(router)

    return app
