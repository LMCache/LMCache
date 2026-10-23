# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory for the mp coordinator.

The coordinator is a FastAPI app. Endpoints are auto-discovered from the
``http_apis`` package (the same convention as the mp server's HTTP API) and stay
thin, operating on the shared collaborators carried on ``app.state``: ``config``,
the view and controller registries, and the ingest layer's ``event_source`` /
``event_gate``.
The lifespan runs health-checking (eviction of instances whose heartbeats have
lapsed) and the checkpoint timer, and starts and stops the event source and
every controller -- this file names no controller of its own.

Adding a capability = a new ``http_apis/<name>_api.py`` router (auto-discovered)
that uses those shared collaborators. A controller that ships outside this tree
cannot be reached from one of those, so it implements ``get_routers`` and brings
its endpoints with it; either way this file names no controller.
"""

# Standard
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
import asyncio
import time

# Third Party
from fastapi import FastAPI
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.background_loops import (
    BackgroundLoops,
    EventLoopMonitor,
)
from lmcache.v1.mp_coordinator.config import (
    KafkaCacheEventSourceConfig,
    MPCoordinatorConfig,
)
from lmcache.v1.mp_coordinator.controllers import build_controllers
from lmcache.v1.mp_coordinator.controllers.base import (
    ControllerRuntime,
    ControllerStatus,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.mp_coordinator.http_routes import HttpRoutes
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import (
    CacheEventBroadcaster,
    CacheEventConsumer,
)
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.event_source import CacheEventSource
from lmcache.v1.mp_coordinator.ingest.http_event_source import HttpCacheEventSource
from lmcache.v1.mp_coordinator.ingest.kafka_event_source import (
    KafkaCacheEventSource,
)
from lmcache.v1.mp_coordinator.ingest.stream_position import StreamPosition
from lmcache.v1.mp_coordinator.observability import register_coordinator_metrics
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
from lmcache.v1.mp_coordinator.views import build_views
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.mp_coordinator.views.server_config import ServerConfigRegistry
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.utils.router_discovery import discover_api_routers

logger = init_logger(__name__)


class StallGrace:
    """Keeps the coordinator's own stalls from timing out the mp servers.

    Heartbeats are only read while the coordinator runs. When it does not
    (event loop blocked, process stopped), what the servers send queues
    unread, and the overdue health pass can run before it. A health pass
    that started more than one interval late shows the coordinator was not
    running from when that pass was due. For ``instance_timeout`` after
    such a pass, liveness is judged as of that due time instead of now:
    an instance already out of time then goes at once, the rest get
    ``instance_timeout`` to heartbeat or register again.

    A stall only shields instances that heartbeat within
    ``instance_timeout`` before its missed pass was due, so recurring
    stalls cannot keep a dead instance registered. See "Coordinator
    stalls" in ``docs/design/v1/mp_coordinator/README.md`` for why one
    interval is the threshold and what it covers.
    """

    def __init__(self, health_check_interval: float, instance_timeout: float) -> None:
        """Create a grace tracker for one health loop.

        Args:
            health_check_interval: Seconds between health passes.
            instance_timeout: Seconds without a heartbeat before an
                instance is evicted; also how long a grace lasts.
        """
        self._interval = health_check_interval
        self._timeout = instance_timeout
        # When the pass the last stall made late was due, and until when
        # liveness is judged as of that moment.
        self._stalled_pass_due = 0.0
        self._grace_until = 0.0

    def observe_pass(self, now: float, late_seconds: float) -> None:
        """Record how late the current health pass started.

        A pass more than one interval late starts a grace of
        ``instance_timeout`` from ``now``, replacing any earlier one, and
        is logged as a WARNING.

        Args:
            now: ``time.monotonic()`` at the start of the pass.
            late_seconds: How much later than due the pass started.
        """
        if late_seconds <= self._interval:
            return
        self._stalled_pass_due = now - late_seconds
        self._grace_until = now + self._timeout
        logger.warning(
            "Coordinator was not running for at least %.1f s; judging "
            "liveness as of then for the next %.1f s so servers can heartbeat",
            late_seconds,
            self._timeout,
        )

    def max_heartbeat_age(self, now: float) -> float:
        """Return the heartbeat age past which an instance is evicted now.

        Args:
            now: ``time.monotonic()`` at the start of the pass.

        Returns:
            ``instance_timeout``, plus the time since the stalled pass was
            due while a grace lasts.
        """
        if now < self._grace_until:
            return self._timeout + now - self._stalled_pass_due
        return self._timeout

    def deadline_passed(self, now: float, deadline: float) -> bool:
        """Return whether a ``time.monotonic()`` deadline has passed.

        While a grace lasts, the deadline counts as passed only if the
        stall surely began after it. The stall began no earlier than one
        interval before the missed pass was due, since each pass is due
        one interval after the previous one finished.

        Args:
            now: ``time.monotonic()`` at the start of the pass.
            deadline: The deadline to check.

        Returns:
            True if the deadline has passed, not counting a stall.
        """
        if now < self._grace_until:
            return self._stalled_pass_due - self._interval >= deadline
        return now >= deadline


def create_app(config: MPCoordinatorConfig) -> FastAPI:
    """Build the coordinator FastAPI app.

    Args:
        config: The coordinator configuration.

    Returns:
        A configured FastAPI application. ``app.state`` carries the shared
        collaborators (``config`` plus the :class:`CoordinatorContext`); all
        ``http_apis`` routers are registered.
    """
    views = build_views(config)
    registry = views.get(InstanceRegistry)
    server_configs = views.get(ServerConfigRegistry)
    controllers = build_controllers(config, views)
    # Resolves pin requests' token_ids to object keys; must match the fleet's
    # chunk size and hash algorithm (see MPCoordinatorConfig).
    token_hasher = TokenHasher(
        chunk_size=config.chunk_size, hash_algorithm=config.hash_algorithm
    )
    # Ingest layer: the source feeds the gate, which admits before the
    # broadcaster fans out. Adding a consumer is a register call here.
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
    # Exactly one source feeds the gate, chosen by the config: the HTTP push
    # source behind ``POST /events`` or the Kafka pull source, never both, so
    # every emitter's stream has one ordered path in (which is what the
    # gate's per-emitter seq cursor assumes).
    event_source: CacheEventSource
    stream_position: StreamPosition | None = None
    if isinstance(config.event_source_config, KafkaCacheEventSourceConfig):
        stream_position = StreamPosition()
        event_source = KafkaCacheEventSource(
            event_gate, config.event_source_config, stream_position
        )
    else:
        event_source = HttpCacheEventSource(event_gate)

    # The gate is named because it is durable but is neither a view nor
    # a controller; everything else advertises its own state. The stream
    # position rides beside them for the same reason a partial checkpoint
    # must never look complete: captured under the same quiesce, restored
    # before the source seeks to it.
    checkpoint_components: list[DurableComponent] = [
        event_gate,
        *views.durable_components()[PersistenceType.CHECKPOINT],
        *controllers.durable_components()[PersistenceType.CHECKPOINT],
    ]
    if stream_position is not None:
        checkpoint_components.append(stream_position)
    checkpoint_store = _artifact_store(config.checkpoint_path)
    metadata_persister = MetadataPersister(_artifact_store(config.metadata_path))
    for component in controllers.durable_components()[PersistenceType.METADATA]:
        metadata_persister.register(component)
    # Before the checkpoint, so a restored key arrives already pinned.
    metadata_persister.load()
    load_checkpoint(checkpoint_store, checkpoint_components)
    background_loops = BackgroundLoops()
    event_loop_monitor = EventLoopMonitor()
    # Whether each controller's run() was entered; the lifespan marks it.
    controller_status = ControllerStatus(type(c).__name__ for c in controllers.all())
    restored_instances = set(event_gate.stats().streams)
    restored_deadline = time.monotonic() + config.instance_timeout
    stall_grace = StallGrace(config.health_check_interval, config.instance_timeout)
    if config.metrics_enabled:
        # Every component with metrics registered itself as it was built.
        register_coordinator_metrics()

    ctx = CoordinatorContext(
        views=views,
        controllers=controllers,
        token_hasher=token_hasher,
        event_gate=event_gate,
        event_source=event_source,
        metadata_persister=metadata_persister,
    )

    async def _write_checkpoint() -> None:
        """One pass of the checkpoint loop.

        Only derived state runs on a timer: it changes continuously, so a
        cadence is the only sensible cost. Operator intent is written when
        it changes instead (see ``MetadataPersister``).
        """
        await asyncio.to_thread(
            save_checkpoint,
            checkpoint_store,
            quiesce,
            checkpoint_components,
        )

    async def _evict_stale_instances() -> None:
        """One pass of the health loop.

        A timed-out instance takes its L1 contents with it, so its
        reported L1 state is fenced across every consumer. Its L2
        contents stay: they live on storage the fleet shares and leave
        only via ``DELETE`` events. Its capacity declaration goes too,
        as on ``DELETE /instances``; a server that was only slow
        redeclares when it re-registers.

        Time the coordinator itself was not running is not held against
        instances or restored ones (see :class:`StallGrace`).
        """
        now = time.monotonic()
        stall_grace.observe_pass(
            now, background_loops.stats()["health"].last_pass_late_seconds
        )
        departed = _evict_stale(registry, stall_grace.max_heartbeat_age(now))
        if restored_instances and stall_grace.deadline_passed(now, restored_deadline):
            departed += _unreturned_instances(registry, restored_instances)
            restored_instances.clear()
        for instance_id in departed:
            event_gate.drop_instance(instance_id)
            server_configs.forget(instance_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start background work and unwind it in order on shutdown.

        Registration order is teardown order reversed, and the order is
        load-bearing: timers stop before the source so no checkpoint races
        ingestion, the source before controllers so no new work arrives while
        they settle, controllers before the final write so it captures what
        they settled on, and the client closes last because a draining
        controller is still using it.

        A controller that raises on the way in is logged and skipped;
        the rest still run.
        """
        async with AsyncExitStack() as stack:
            # Bound to the running event loop, so it cannot be built with
            # the rest of the app.
            outbound_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=30.0)
            )
            app.state.outbound_client = outbound_client
            if config.checkpoint_path:
                # One last write on the way out, so a clean restart
                # resumes here rather than at an interval-old copy.
                stack.push_async_callback(
                    asyncio.to_thread,
                    save_checkpoint,
                    checkpoint_store,
                    quiesce,
                    checkpoint_components,
                )
            # One controller is not allowed to take the coordinator down
            # with it: the endpoints belonging to no controller keep
            # working, and whatever the failed one does simply is not
            # happening -- the log is the only notice of that.
            runtime = ControllerRuntime(
                http_client=outbound_client, background_loops=background_loops
            )
            for controller in controllers.all():
                try:
                    await stack.enter_async_context(controller.run(runtime))
                except Exception:
                    logger.exception(
                        "Controller %s failed to start", type(controller).__name__
                    )
                    continue
                controller_status.mark_up(type(controller).__name__)
            await event_source.start()
            stack.push_async_callback(event_source.stop)
            # Nested, so they stop before the stack unwinds. Awaited too:
            # ``save_checkpoint`` runs in a thread a cancel cannot reach.
            timers = [asyncio.create_task(event_loop_monitor.run())]
            if config.checkpoint_path and config.checkpoint_interval > 0:
                timers.append(
                    background_loops.start(
                        "checkpoint", config.checkpoint_interval, _write_checkpoint
                    )
                )
            if config.health_check_interval > 0:
                timers.append(
                    background_loops.start(
                        "health", config.health_check_interval, _evict_stale_instances
                    )
                )
            logger.info(
                "MP coordinator listening on http://%s:%d", config.host, config.port
            )
            try:
                yield
            finally:
                for timer in timers:
                    timer.cancel()
                await asyncio.gather(*timers, return_exceptions=True)

    app = FastAPI(title="LMCache MP Coordinator", version="1.0.0", lifespan=lifespan)
    app.state.ctx = ctx
    # Out-of-context collaborator kept on app.state directly.
    app.state.config = config

    apis_path = Path(__file__).parent / "http_apis"
    package = f"{__package__}.http_apis"
    for router in discover_api_routers(apis_path, package):
        app.include_router(router)
    # Then whatever a controller brings itself, so one this file cannot
    # name still gets its endpoints. In-tree routes are mounted above, so
    # they win a path collision.
    for member in (*views.all(), *controllers.all()):
        if isinstance(member, HttpRoutes):
            for router in member.get_routers():
                app.include_router(router)

    return app


def _artifact_store(path: str) -> ArtifactStore:
    """Return the store for ``path``, or one that discards if unset.

    Args:
        path: Configured location, empty when the operator wants none.
    """
    return LocalArtifactStore(Path(path)) if path else NullArtifactStore()


def _evict_stale(registry: InstanceRegistry, max_age: float) -> list[str]:
    """Deregister every instance whose heartbeat is older than ``max_age``.

    Args:
        registry: The shared instance registry.
        max_age: Heartbeat age, in seconds, this sweep allows.

    Returns:
        The ids of instances evicted in this sweep.
    """
    evicted = []
    for instance_id in registry.stale(max_age):
        if registry.deregister(instance_id) is not None:
            logger.warning("Instance %s timed out; evicted", instance_id)
            evicted.append(instance_id)
    return evicted


def _unreturned_instances(
    registry: InstanceRegistry, restored: Iterable[str]
) -> list[str]:
    """Return the restored instances that never registered again.

    Args:
        registry: The shared instance registry.
        restored: Instances that came back from the checkpoint.

    Returns:
        The ids still missing from the registry.
    """
    missing = []
    for instance_id in restored:
        if not registry.contains(instance_id):
            logger.warning(
                "Instance %s did not re-register after restore; evicted", instance_id
            )
            missing.append(instance_id)
    return missing
