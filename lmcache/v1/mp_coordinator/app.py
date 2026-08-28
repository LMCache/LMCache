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
from lmcache.v1.mp_coordinator.controllers import build_controllers
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import (
    CacheEventBroadcaster,
    CacheEventConsumer,
)
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactStore,
    LocalArtifactStore,
    NullArtifactStore,
)
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.mp_coordinator.views import build_views
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
    views = build_views(config)
    controllers = build_controllers(config, views)
    eviction_controller = controllers.get(FleetEvictionController)
    # Resolves pin requests' token_ids to object keys; must match the fleet's
    # chunk size and hash algorithm (see MPCoordinatorConfig).
    token_hasher = TokenHasher(
        chunk_size=config.chunk_size, hash_algorithm=config.hash_algorithm
    )
    # Ingest layer: the gate admits, the broadcaster fans out. Adding a
    # consumer of the fleet's cache-event stream is a register call here.
    event_broadcaster = CacheEventBroadcaster()
    # Views first: a controller acts on the batch a view has consumed.
    # Not everything discovered consumes, so the protocol decides.
    for collaborator in (*views.all(), *controllers.all()):
        if isinstance(collaborator, CacheEventConsumer):
            event_broadcaster.register_consumer(collaborator)
    # Held by the ingest path; whoever captures durable state takes it
    # to read across the consumers consistently.
    quiesce = QuiesceLock()
    event_gate = EventGate(event_broadcaster, quiesce)

    # The gate is named because it is durable but is neither a view nor
    # a controller; everything else advertises its own state.
    checkpoint_components: list[DurableComponent] = [
        event_gate,
        *views.durable_components()[PersistenceType.CHECKPOINT],
        *controllers.durable_components()[PersistenceType.CHECKPOINT],
    ]
    checkpoint_store = _artifact_store(config.checkpoint_path)
    metadata_persister = MetadataPersister(_artifact_store(config.metadata_path))
    for component in controllers.durable_components()[PersistenceType.METADATA]:
        metadata_persister.register(component)
    # Before the checkpoint, so a restored key arrives already pinned.
    metadata_persister.load()
    load_checkpoint(checkpoint_store, checkpoint_components)

    ctx = CoordinatorContext(
        registry=registry,
        views=views,
        controllers=controllers,
        token_hasher=token_hasher,
        event_gate=event_gate,
        metadata_persister=metadata_persister,
    )

    async def _checkpoint_loop() -> None:
        """Checkpoint on a timer until cancelled.

        Only derived state runs on a timer: it changes continuously, so a
        cadence is the only sensible cost. Operator intent is written when
        it changes instead (see ``MetadataPersister``).
        """
        while True:
            await asyncio.sleep(config.checkpoint_interval)
            await asyncio.to_thread(
                save_checkpoint,
                checkpoint_store,
                quiesce,
                checkpoint_components,
            )

    async def _health_loop() -> None:
        """Evict stale instances on a timer until cancelled.

        A timed-out instance takes its L1 contents with it, so its
        reported L1 state is fenced across every consumer. Its L2
        contents stay: they live on storage the fleet shares and leave
        only via ``DELETE`` events.
        """
        while True:
            await asyncio.sleep(config.health_check_interval)
            for instance_id in evict_stale(registry, config.instance_timeout):
                event_gate.drop_instance(instance_id)

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
        checkpoint_task = None
        if config.checkpoint_path and config.checkpoint_interval > 0:
            checkpoint_task = asyncio.create_task(_checkpoint_loop())
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
            for task in (health_task, eviction_task, checkpoint_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if config.checkpoint_path:
                # One last write, so a clean restart resumes where this
                # process stopped rather than an interval-old copy.
                await asyncio.to_thread(
                    save_checkpoint,
                    checkpoint_store,
                    quiesce,
                    checkpoint_components,
                )
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


def _artifact_store(path: str) -> ArtifactStore:
    """Return the store for ``path``, or one that discards if unset.

    Args:
        path: Configured location, empty when the operator wants none.
    """
    return LocalArtifactStore(Path(path)) if path else NullArtifactStore()
