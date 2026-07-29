# SPDX-License-Identifier: Apache-2.0
"""MP-server-side cache-event emission for the coordinator key directory.

Storage listeners (:class:`L1CacheEventListener`, per-adapter
:class:`L2CacheEventListener`) feed a :class:`CacheEventEmitter`, which
flushes ordered :class:`CacheEventBatch` lists through a
:class:`CacheEventSink` — the transport seam (HTTP today, a message
queue later). See ``docs/design/v1/mp_coordinator/cache_events.md``.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
import threading
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1Backend, ObjectKey, Tier
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1ObjectMeta,
    L2AdapterListener,
)
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.schemas import DirectoryEventsRequest

logger = init_logger(__name__)

_DEFAULT_FLUSH_INTERVAL = 1.0


class CacheEventPublishError(Exception):
    """A sink failed to deliver a list of cache-event batches."""


class CacheEventSink(ABC):
    """Transport seam for delivering cache-event batches to the directory.

    Implementations provide at-least-once delivery and preserve batch
    order within and across :meth:`publish` calls; the directory's seq
    dedup, gap detection, and incarnation fencing absorb everything else.
    """

    @abstractmethod
    async def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` to the directory, in list order.

        Args:
            batches: The batches to deliver; never empty.

        Raises:
            CacheEventPublishError: If delivery failed. Retrying and
                dropping are both safe (dedup / gap-flagged resync).
        """
        raise NotImplementedError


class HttpCacheEventSink(CacheEventSink):
    """Sink that POSTs batches to the coordinator's ``/directory/events``.

    Args:
        client: HTTP client to send with (owned by the caller).
        coordinator_url: Coordinator base URL.
    """

    def __init__(self, client: httpx.AsyncClient, coordinator_url: str) -> None:
        self._client = client
        self._base_url = coordinator_url.rstrip("/")

    async def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` via one ``POST /directory/events`` request.

        Args:
            batches: The batches to deliver; never empty.

        Raises:
            CacheEventPublishError: If the request failed or returned
                a non-2xx status.
        """
        body = DirectoryEventsRequest(batches=batches)
        try:
            resp = await self._client.post(
                f"{self._base_url}/directory/events",
                json=body.model_dump(mode="json"),
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise CacheEventPublishError(
                f"failed to publish {len(batches)} cache-event batches to "
                f"{self._base_url}: {e}"
            ) from e


@dataclass
class _PendingRun:
    """One buffered run of same-identity events awaiting flush."""

    event_type: CacheEventType
    tier: Tier
    backend: str
    entries: list[CacheEventEntry]


class CacheEventEmitter:
    """Buffers cache events on the MP server and flushes ordered batches.

    Listeners call :meth:`record` from any thread; :meth:`run` flushes on
    a timer. Consecutive records with the same ``(event_type, tier,
    backend)`` identity coalesce into one batch, preserving the total
    order of recorded events. Failed flushes drop their batches but keep
    the consumed ``seq`` numbers, so the loss surfaces as a gap that
    flags this instance for resync.

    Args:
        sink: Transport that delivers flushed batches.
        instance_id: This MP server's id (sent with every batch).
        incarnation: This server process's incarnation (its start time);
            fences out placements reported before a restart.
        flush_interval: Seconds between flush attempts.
    """

    def __init__(
        self,
        sink: CacheEventSink,
        instance_id: str,
        incarnation: int,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self._sink = sink
        self._instance_id = instance_id
        self._incarnation = incarnation
        self._flush_interval = flush_interval
        self._seq = 0
        self._lock = threading.Lock()
        # Pending runs: consecutive same-identity records share one run.
        self._runs: list[_PendingRun] = []

    def record(
        self,
        event_type: CacheEventType,
        tier: Tier,
        backend: str,
        entries: list[CacheEventEntry],
    ) -> None:
        """Buffer ``entries`` for the next flush. Thread-safe.

        Args:
            event_type: What happened to every entry.
            tier: The cache tier the events apply to (``l1`` or ``l2``).
            backend: The storage backend within the tier (non-empty).
            entries: The affected keys; an empty list is a no-op.
        """
        if not entries:
            return
        with self._lock:
            if self._runs:
                last = self._runs[-1]
                if (
                    last.event_type == event_type
                    and last.tier == tier
                    and last.backend == backend
                ):
                    last.entries.extend(entries)
                    return
            self._runs.append(
                _PendingRun(
                    event_type=event_type,
                    tier=tier,
                    backend=backend,
                    entries=list(entries),
                )
            )

    async def run(self) -> None:
        """Flush the buffer on a timer until cancelled."""
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.flush()

    async def flush(self) -> None:
        """Drain the buffer and publish one batch per pending run.

        Publish failures are logged and the drained list is dropped.
        """
        with self._lock:
            if not self._runs:
                return
            runs = self._runs
            self._runs = []
            ts = time.time()
            batches = [
                CacheEventBatch(
                    instance_id=self._instance_id,
                    incarnation=self._incarnation,
                    seq=self._seq + offset + 1,
                    event_type=run.event_type,
                    tier=run.tier,
                    backend=run.backend,
                    entries=run.entries,
                    ts=ts,
                )
                for offset, run in enumerate(runs)
            ]
            self._seq += len(runs)
        try:
            await self._sink.publish(batches)
        except CacheEventPublishError as e:
            logger.warning(
                "Dropping %d cache-event batches (instance %s): %s",
                len(batches),
                self._instance_id,
                e,
            )


def l1_backend_name(config: L1ManagerConfig) -> L1Backend:
    """Return the primary backend id for the configured L1 medium.

    Used to label recency (``ACCESS``) events; placement-bearing events
    carry the per-object medium from the listener callbacks instead.

    Args:
        config: The L1 manager configuration.

    Returns:
        The configured primary medium (hybrid DRAM+DAX reports
        ``DEVDAX``).
    """
    if config.gds_l1_config is not None:
        return L1Backend.GDS
    if config.memory_config.devdax_path:
        return L1Backend.DEVDAX
    return L1Backend.DRAM


class L1CacheEventListener(L1ManagerListener):
    """L1 manager listener that forwards events to a directory emitter.

    Writes (including prefetch completions) map to ``STORE`` and manager
    deletions to ``DELETE``, each recorded under the per-object medium
    from the callback metadata — a hybrid DRAM+DAX L1 emits per-medium
    batches, and a delete targets the same placement identity its store
    reported. Reads and touches map to ``ACCESS`` under the configured
    primary medium (recency never creates placements, so the label is
    cosmetic); reservations are ignored. All callbacks are thread-safe.

    Args:
        emitter: The emitter to forward events to.
        access_backend: Label for ``ACCESS`` batches (see
            :func:`l1_backend_name`).
    """

    def __init__(self, emitter: CacheEventEmitter, access_backend: L1Backend) -> None:
        self._emitter = emitter
        self._access_backend = access_backend.value

    # -- L1ManagerListener implementation -------------------------------------

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """Ignore read reservations (not a state change)."""

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Record an ``ACCESS`` event per read key."""
        self._record_access(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """Ignore write reservations (data not committed yet)."""

    def on_l1_keys_write_finished(
        self, keys: list[ObjectKey], metadata: list[L1ObjectMeta]
    ) -> None:
        """Record a ``STORE`` event per written key, split by medium."""
        self._record_placements(CacheEventType.STORE, keys, metadata)

    def on_l1_keys_finish_write_and_reserve_read(
        self, keys: list[ObjectKey], metadata: list[L1ObjectMeta]
    ) -> None:
        """Record a ``STORE`` event per prefetched key, split by medium."""
        self._record_placements(CacheEventType.STORE, keys, metadata)

    def on_l1_keys_deleted_by_manager(
        self, keys: list[ObjectKey], metadata: list[L1ObjectMeta]
    ) -> None:
        """Record a ``DELETE`` event per deleted key, split by medium
        (evictions included)."""
        self._record_placements(CacheEventType.DELETE, keys, metadata)

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Record an ``ACCESS`` event per touched key."""
        self._record_access(keys)

    # -- Internals -------------------------------------------------------------

    def _record_placements(
        self,
        event_type: CacheEventType,
        keys: list[ObjectKey],
        metadata: list[L1ObjectMeta],
    ) -> None:
        """Record one ``event_type`` run per medium found in ``metadata``."""
        by_backend: dict[L1Backend, list[CacheEventEntry]] = {}
        for key, meta in zip(keys, metadata, strict=True):
            by_backend.setdefault(meta.backend, []).append(
                CacheEventEntry(
                    key=key.to_encoded_object_key(),
                    size_bytes=meta.size_bytes
                    if event_type is CacheEventType.STORE
                    else 0,
                )
            )
        for backend, entries in by_backend.items():
            self._emitter.record(event_type, Tier.L1, backend.value, entries)

    def _record_access(self, keys: list[ObjectKey]) -> None:
        """Record an ``ACCESS`` event per key under the primary medium."""
        self._emitter.record(
            CacheEventType.ACCESS,
            Tier.L1,
            self._access_backend,
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )


class L2CacheEventListener(L2AdapterListener):
    """L2 adapter listener that forwards events to a directory emitter.

    Register one instance per adapter so events carry that adapter's
    backend name (the callbacks do not identify the emitting adapter).
    All callbacks are thread-safe.

    Args:
        emitter: The emitter to forward events to.
        backend: The adapter's backend name (e.g. ``"fs"``, ``"valkey"``).
    """

    def __init__(self, emitter: CacheEventEmitter, backend: str) -> None:
        self._emitter = emitter
        self._backend = backend

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Record a ``STORE`` event per stored key."""
        self._emitter.record(
            CacheEventType.STORE,
            Tier.L2,
            self._backend,
            [
                CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size)
                for key, size in zip(keys, sizes, strict=True)
            ],
        )

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Record an ``ACCESS`` event per accessed key."""
        self._emitter.record(
            CacheEventType.ACCESS,
            Tier.L2,
            self._backend,
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        """Record a ``DELETE`` event per deleted key."""
        self._emitter.record(
            CacheEventType.DELETE,
            Tier.L2,
            self._backend,
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )
