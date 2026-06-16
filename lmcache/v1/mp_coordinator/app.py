# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory for the mp coordinator.

The coordinator is a FastAPI app. Endpoints are auto-discovered from the
``http_apis`` package (the same convention as the mp server's HTTP API) and stay
thin, operating on the shared collaborators carried on ``app.state``: ``config``,
``registry``, ``quota_manager``, ``usage_manager``, and ``eviction_manager``.
The lifespan runs background tasks for health-checking (eviction of instances
whose heartbeats have lapsed) and L2 eviction (quota enforcement).

Adding a capability = a new ``http_apis/<name>_api.py`` router (auto-discovered)
that uses those shared collaborators. To push to an mp server, a future router
resolves the instance's address from the registry (``ip`` + ``http_port``) and
POSTs to that server's specific endpoint. A domain with real logic/state of its
own adds a ``<name>_service.py`` stashed on ``app.state`` here; thin domains
(like membership) just use the registry directly.
"""

# Standard
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import contextlib

# Third Party
from fastapi import FastAPI

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.blend_directory import GlobalBlendMatcher
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.l2.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
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
        collaborators (``config``, ``registry``, ``quota_manager``,
        ``usage_manager``, ``blend_directory``); all ``http_apis`` routers are
        registered.
    """
    registry = InstanceRegistry()
    quota_manager = QuotaManager()
    usage_manager = L2UsageManager()
    eviction_manager = L2EvictionManager(
        quota_manager=quota_manager,
        usage_manager=usage_manager,
        eviction_ratio=config.eviction_ratio,
        trigger_watermark=config.trigger_watermark,
    )
    blend_directory = GlobalBlendMatcher(
        chunk_size=config.blend_chunk_size, probe_stride=config.blend_probe_stride
    )

    async def _health_loop() -> None:
        """Evict stale instances on a timer until cancelled."""
        while True:
            await asyncio.sleep(config.health_check_interval)
            evict_stale(registry, config.instance_timeout)

    async def _eviction_loop() -> None:
        """Periodically check usage against quotas and log eviction plans."""
        while True:
            await asyncio.sleep(config.eviction_check_interval)
            eviction_manager.execute_evictions()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start background tasks and clean up resources on shutdown."""
        health_task = None
        eviction_task = None
        if config.health_check_interval > 0:
            health_task = asyncio.create_task(_health_loop())
        if config.eviction_check_interval > 0:
            eviction_task = asyncio.create_task(_eviction_loop())
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

    app = FastAPI(title="LMCache MP Coordinator", version="1.0.0", lifespan=lifespan)
    # Shared collaborators on app.state so routers compose from them.
    app.state.config = config
    app.state.registry = registry
    app.state.quota_manager = quota_manager
    app.state.usage_manager = usage_manager
    app.state.eviction_manager = eviction_manager
    app.state.blend_directory = blend_directory

    apis_path = Path(__file__).parent / "http_apis"
    package = f"{__package__}.http_apis"
    for router in discover_api_routers(apis_path, package):
        app.include_router(router)

    return app
