# SPDX-License-Identifier: Apache-2.0
"""Replaying an ``events``-level trace into a coordinator.

``lmcache server --trace-level events`` writes one file per server process
holding the cache-event stream in wire form (see
:mod:`lmcache.v1.mp_coordinator.cache_events`). A fleet is a set of such
files. :meth:`EventsTrace.load` merges them into one stream ordered by
wall-clock time, and :func:`replay` feeds that stream to a
:class:`CoordinatorTarget`, which registers and reports over HTTP the way a
live server does, so a recorded fleet can be replayed into a coordinator
that never saw it.

Batches are handed over exactly as recorded, so a file has the same
compatibility with a newer coordinator that a live server would have. Only
``seq`` order within a server matters to the coordinator; the merge across
servers is by each host's wall clock and carries that clock's skew.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import asyncio
import contextlib
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import CACHE_EVENT_SCHEMA_VERSION
from lmcache.v1.mp_coordinator.cache_events import (
    EVENTS_TRACE_BATCH,
    EVENTS_TRACE_LIFECYCLE,
    TraceLifecyclePhase,
)
from lmcache.v1.mp_coordinator.registrar import register
from lmcache.v1.mp_coordinator.schemas import CacheEventsResponse
from lmcache.v1.mp_observability.trace.format import Record
from lmcache.v1.mp_observability.trace.lifecycle import EVENTS_LEVEL
from lmcache.v1.mp_observability.trace.reader import TraceReader

logger = init_logger(__name__)

_LOCALHOST = "127.0.0.1"
_DEFAULT_HEARTBEAT_INTERVAL = 5.0
"""The MP server's own default, well under the coordinator's 30 s timeout."""


@dataclass(frozen=True)
class EmitterIdentity:
    """What a ``start`` mark says about the server that wrote the file.

    Attributes:
        instance_id: The server's id.
        incarnation: The incarnation that began at the mark.
        ip: The address the server advertised, empty when it deferred to
            its outbound address.
        http_port: Its HTTP port; ``0`` when it ran no HTTP frontend.
        mq_port: Its message-queue port; ``0`` when P2P was off.
    """

    instance_id: str
    incarnation: int
    ip: str = ""
    http_port: int = 0
    mq_port: int = 0

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> EmitterIdentity:
        """Read the identity off a ``start`` mark's ``args``.

        Args:
            args: The record's arguments as read from the file.

        Returns:
            The identity; fields the mark lacks take their defaults.
        """
        return cls(
            instance_id=str(args.get("instance_id", "")),
            incarnation=int(args.get("incarnation", 0)),
            ip=str(args.get("ip", "")),
            http_port=int(args.get("http_port", 0)),
            mq_port=int(args.get("mq_port", 0)),
        )


@dataclass(frozen=True)
class Ingested:
    """How a coordinator counted replayed batches.

    Attributes:
        applied: Batches admitted.
        duplicates: Batches dropped as already applied.
        stale: Batches dropped for an outdated incarnation.
    """

    applied: int = 0
    duplicates: int = 0
    stale: int = 0

    def __add__(self, other: Ingested) -> Ingested:
        return Ingested(
            applied=self.applied + other.applied,
            duplicates=self.duplicates + other.duplicates,
            stale=self.stale + other.stale,
        )


@dataclass
class EventsTrace:
    """A fleet's cache-event stream, one record after another in time order.

    Attributes:
        records: Batches and lifecycle marks, ascending by ``t_wall``.
            Records with equal timestamps keep file order, so no server's
            sequence is reordered.
        meta: The ``level_meta`` of the first file loaded.
        files: The files the records came from.
    """

    records: list[Record] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, paths: Sequence[str]) -> EventsTrace:
        """Read one or more ``events``-level files into one stream.

        Args:
            paths: The files, typically one per recorded server.

        Returns:
            Their records merged by wall-clock time.

        Raises:
            ValueError: If a file is not an ``events``-level trace, or its
                batches use a cache-event schema this build does not speak.
        """
        records: list[Record] = []
        meta: dict[str, Any] = {}
        for path in paths:
            with TraceReader(path) as reader:
                header = reader.header
                if header.level != EVENTS_LEVEL:
                    raise ValueError(
                        f"trace {path!r} is level {header.level!r}; only an "
                        f"{EVENTS_LEVEL!r} trace holds coordinator input"
                    )
                schema = header.level_meta.get("cache_event_schema_version")
                if schema is not None and schema != CACHE_EVENT_SCHEMA_VERSION:
                    raise ValueError(
                        f"trace {path!r} holds cache_event_schema_version "
                        f"{schema}; this build speaks {CACHE_EVENT_SCHEMA_VERSION}"
                    )
                if not meta:
                    meta = dict(header.level_meta)
                records.extend(reader.records())
        records.sort(key=lambda record: record.t_wall)
        return cls(records=records, meta=meta, files=list(paths))

    @property
    def instances(self) -> list[str]:
        """The servers named in the stream, in first-appearance order."""
        seen: dict[str, None] = {}
        for record in self.records:
            instance_id = str(record.args.get("instance_id", ""))
            if instance_id:
                seen.setdefault(instance_id, None)
        return list(seen)


class CoordinatorTarget:
    """Delivers the stream to a coordinator over HTTP, as a server would.

    A ``start`` mark registers the instance (``POST /instances``, at the
    address the mark recorded or loopback when it recorded none) and begins
    heartbeating it on a timer, a batch is one ``POST /events``, a ``stop``
    mark deregisters. A stream that ends without ``stop`` just stops
    heartbeating, and the coordinator's health loop retires the server after
    its ``instance_timeout``, the way it learns of a real crash. Heartbeats
    end when the target closes; use it as an async context manager.

    The coordinator must not have the same fleet live: registering a
    recorded server replaces a live one's registration.

    Args:
        client: The client to send with.
        coordinator_url: The coordinator's base URL.
        heartbeat_interval: Seconds between heartbeats of a registered
            server; must stay below the coordinator's ``instance_timeout``.
            ``0`` sends none.

    Raises:
        ValueError: If ``heartbeat_interval`` is negative.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        coordinator_url: str,
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        if heartbeat_interval < 0:
            raise ValueError(
                f"heartbeat_interval must be >= 0 (got {heartbeat_interval})"
            )
        self._client = client
        self._base_url = coordinator_url.rstrip("/")
        self._heartbeat_interval = heartbeat_interval
        self._registered: set[str] = set()
        self._heartbeats: dict[str, asyncio.Task[None]] = {}

    async def __aenter__(self) -> CoordinatorTarget:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop every heartbeat. Registrations are left for the coordinator
        to retire, as a fleet that went away would be."""
        for instance_id in list(self._heartbeats):
            await self._stop_heartbeat(instance_id)

    async def start(self, identity: EmitterIdentity) -> None:
        """A server began emitting: register it and begin heartbeating it,
        unless the mark recorded no HTTP port. A second call for the same
        id with a higher incarnation is a restart and registers again.

        Args:
            identity: Who began, as the ``start`` mark recorded it.

        Raises:
            httpx.HTTPError: If the coordinator refused the registration.
        """
        if identity.http_port <= 0:
            logger.warning(
                "events replay: %s recorded no HTTP port; not registering it",
                identity.instance_id,
            )
            return
        await register(
            self._client,
            self._base_url,
            http_port=identity.http_port,
            advertise_ip=identity.ip or _LOCALHOST,
            instance_id=identity.instance_id,
            mq_port=identity.mq_port,
        )
        self._registered.add(identity.instance_id)
        if (
            self._heartbeat_interval > 0
            and identity.instance_id not in self._heartbeats
        ):
            self._heartbeats[identity.instance_id] = asyncio.create_task(
                self._heartbeat(identity.instance_id)
            )

    async def batch(self, batch: dict[str, Any]) -> Ingested:
        """``POST /events`` with one batch.

        Args:
            batch: The batch in wire form, one element of a
                ``POST /events`` body, exactly as recorded.

        Returns:
            How the coordinator counted it.

        Raises:
            httpx.HTTPError: If the request failed or was refused.
        """
        response = await self._client.post(
            f"{self._base_url}/events", json={"batches": [batch]}
        )
        response.raise_for_status()
        counts = CacheEventsResponse.model_validate(response.json())
        return Ingested(
            applied=counts.applied, duplicates=counts.duplicates, stale=counts.stale
        )

    async def stop(self, instance_id: str) -> None:
        """A server shut down cleanly: stop heartbeating and deregister it,
        if this target registered it.

        Args:
            instance_id: Who stopped.

        Raises:
            httpx.HTTPError: If the coordinator refused the call.
        """
        await self._stop_heartbeat(instance_id)
        if instance_id not in self._registered:
            return
        response = await self._client.delete(
            f"{self._base_url}/instances/{instance_id}"
        )
        response.raise_for_status()
        self._registered.discard(instance_id)

    async def end(self, instance_id: str) -> None:
        """A server's stream ended with no ``stop`` mark: it died, or its
        recording was cut. Stop heartbeating it and leave it registered, so
        the coordinator's health loop retires it by its own timeout.

        Args:
            instance_id: Whose stream ended.
        """
        await self._stop_heartbeat(instance_id)

    async def _heartbeat(self, instance_id: str) -> None:
        """``PUT /instances/{id}/heartbeat`` every interval until cancelled.

        A failed call is logged and retried next tick, like a live server's.
        A 404 means the coordinator forgot the server; the loop ends, since
        re-registering would revive what the coordinator decided was gone.
        """
        url = f"{self._base_url}/instances/{instance_id}/heartbeat"
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                response = await self._client.put(url)
            except httpx.HTTPError as e:
                logger.warning(
                    "events replay: heartbeat of %s failed: %s", instance_id, e
                )
                continue
            if response.status_code == 404:
                logger.warning(
                    "events replay: coordinator no longer knows %s; heartbeats end",
                    instance_id,
                )
                self._registered.discard(instance_id)
                return

    async def _stop_heartbeat(self, instance_id: str) -> None:
        """Cancel the server's heartbeat task, if any, and wait for it."""
        task = self._heartbeats.pop(instance_id, None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@dataclass
class EventsReplayResult:
    """What one :func:`replay` did.

    Attributes:
        records: Records read.
        batches: Batches delivered.
        starts: ``start`` marks delivered.
        stops: ``stop`` marks delivered.
        ended: Streams that ended with no ``stop`` mark: servers that died,
            or recordings that were cut.
        skipped: Records of a kind the replayer does not know, or marks
            naming no instance.
        ingested: The coordinator's counts over every batch.
        duration_s: Wall-clock seconds the replay took.
    """

    records: int = 0
    batches: int = 0
    starts: int = 0
    stops: int = 0
    ended: int = 0
    skipped: int = 0
    ingested: Ingested = field(default_factory=Ingested)
    duration_s: float = 0.0


async def replay(
    trace: EventsTrace, target: CoordinatorTarget, speed: float = 0.0
) -> EventsReplayResult:
    """Deliver every record of ``trace`` to ``target``, in order, logging
    each one.

    Args:
        trace: The stream.
        target: Where to deliver it.
        speed: Pacing relative to the recording's wall clock: ``1.0``
            replays at the recorded rate, ``2.0`` twice as fast, ``0``
            (the default) as fast as the target takes records. The
            coordinator orders by ``seq``, so collapsing the gaps is safe.

    Returns:
        Counts of what was delivered.

    Raises:
        ValueError: If ``speed`` is negative.
    """
    if speed < 0:
        raise ValueError(f"speed must be >= 0 (got {speed})")
    result = EventsReplayResult()
    started = time.monotonic()
    origin = trace.records[0].t_wall if trace.records else 0.0
    last_record = _last_record_per_instance(trace.records)
    total = len(trace.records)
    for index, record in enumerate(trace.records):
        if speed > 0:
            due = started + (record.t_wall - origin) / speed
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        ingested = await _deliver(record, target, result)
        result.records += 1
        instance_id = str(record.args.get("instance_id", ""))
        logger.info(
            "[%d/%d] %s %s%s",
            index + 1,
            total,
            record.qualname,
            instance_id,
            "" if ingested is None else f" {ingested}",
        )
        if last_record.get(instance_id) == index and not _is_stop(record):
            await target.end(instance_id)
            result.ended += 1
    result.duration_s = time.monotonic() - started
    return result


async def _deliver(
    record: Record, target: CoordinatorTarget, result: EventsReplayResult
) -> Ingested | None:
    """Hand one record to the target and count it."""
    if record.qualname == EVENTS_TRACE_BATCH:
        ingested = await target.batch(record.args)
        result.batches += 1
        result.ingested = result.ingested + ingested
        return ingested
    if record.qualname != EVENTS_TRACE_LIFECYCLE:
        result.skipped += 1
        logger.warning("events replay: skipping unknown record %r", record.qualname)
        return None
    phase = str(record.args.get("phase", ""))
    instance_id = str(record.args.get("instance_id", ""))
    if not instance_id:
        result.skipped += 1
        logger.warning("events replay: skipping a %r mark naming no instance", phase)
        return None
    if phase == TraceLifecyclePhase.START.value:
        await target.start(EmitterIdentity.from_args(record.args))
        result.starts += 1
    elif phase == TraceLifecyclePhase.STOP.value:
        await target.stop(instance_id)
        result.stops += 1
    else:
        result.skipped += 1
        logger.warning("events replay: skipping unknown lifecycle phase %r", phase)
    return None


def _last_record_per_instance(records: Sequence[Record]) -> dict[str, int]:
    """The index of each instance's final record, to tell a stream that ends
    without a ``stop`` mark from one that is still going."""
    last: dict[str, int] = {}
    for index, record in enumerate(records):
        instance_id = str(record.args.get("instance_id", ""))
        if instance_id:
            last[instance_id] = index
    return last


def _is_stop(record: Record) -> bool:
    """Whether ``record`` is a ``stop`` mark."""
    return (
        record.qualname == EVENTS_TRACE_LIFECYCLE
        and record.args.get("phase") == TraceLifecyclePhase.STOP.value
    )
