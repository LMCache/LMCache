# SPDX-License-Identifier: Apache-2.0
"""Tests for replaying ``events``-level traces into a coordinator.

Files are written with the real ``EventsTraceRecorder`` and read back
through ``EventsTrace``; the coordinator is the real app behind an
in-process ASGI transport. Everything runs on CPU.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import asyncio
import time

# Third Party
from fastapi import FastAPI
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CACHE_EVENT_SCHEMA_VERSION,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app, evict_stale
from lmcache.v1.mp_coordinator.cache_events import (
    EVENTS_TRACE_BATCH,
    EVENTS_TRACE_LIFECYCLE,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.events_replay import (
    CoordinatorTarget,
    EmitterIdentity,
    EventsTrace,
    Ingested,
    replay,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.recorder import (
    EventsTraceRecorder,
    StorageTraceRecorder,
)


@pytest.fixture(autouse=True)
def _reset_trace_gate():
    yield
    set_tracing_enabled(False)


def _key(chunk: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk), model_name="m", kv_rank=0, cache_salt="alice"
    )


def _batch(instance_id: str, incarnation: int, seq: int, chunk: str = "aa") -> dict:
    """One STORE batch in wire form, as the recorder holds it."""
    batch = CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[
            CacheEventEntry(key=_key(chunk).to_encoded_object_key(), size_bytes=10)
        ],
    )
    return CacheEventsRequest(batches=[batch]).model_dump(mode="json")["batches"][0]


def _start(instance_id: str, incarnation: int, http_port: int = 8000) -> dict:
    return {
        "phase": "start",
        "instance_id": instance_id,
        "incarnation": incarnation,
        "ip": "",
        "http_port": http_port,
        "mq_port": 0,
    }


def _stop(instance_id: str) -> dict:
    return {"phase": "stop", "instance_id": instance_id}


def _write(
    path: str,
    records: list[tuple[float, str, dict[str, Any]]],
    schema: int = CACHE_EVENT_SCHEMA_VERSION,
) -> str:
    """Write an events file with the given ``(t_wall, qualname, args)`` records."""
    recorder = EventsTraceRecorder(
        path, level_meta={"cache_event_schema_version": schema}
    )
    for t_wall, qualname, args in records:
        recorder.write_record(qualname, args, t_wall=t_wall, t_mono=t_wall)
    recorder.close()
    return path


class _RecordingTarget(CoordinatorTarget):
    """Remembers every delivery, in order, and talks to no coordinator."""

    def __init__(self) -> None:
        super().__init__(httpx.AsyncClient(), "http://nowhere", heartbeat_interval=0)
        self.calls: list[tuple[str, object]] = []

    async def start(self, identity: EmitterIdentity) -> None:
        self.calls.append(("start", identity))

    async def batch(self, batch: dict[str, Any]) -> Ingested:
        self.calls.append(("batch", (batch["instance_id"], batch["seq"])))
        return Ingested(applied=1)

    async def stop(self, instance_id: str) -> None:
        self.calls.append(("stop", instance_id))

    async def end(self, instance_id: str) -> None:
        self.calls.append(("end", instance_id))


# -- Loading -------------------------------------------------------------------


def test_load_merges_files_by_wall_clock(tmp_path):
    a = _write(
        str(tmp_path / "a.lct"),
        [
            (1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1)),
            (2.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
            (4.0, EVENTS_TRACE_LIFECYCLE, _stop("node-a")),
        ],
    )
    b = _write(
        str(tmp_path / "b.lct"),
        [
            (1.5, EVENTS_TRACE_LIFECYCLE, _start("node-b", 1)),
            (3.0, EVENTS_TRACE_BATCH, _batch("node-b", 1, 1)),
        ],
    )

    trace = EventsTrace.load([a, b])

    assert [(r.t_wall, r.args["instance_id"]) for r in trace.records] == [
        (1.0, "node-a"),
        (1.5, "node-b"),
        (2.0, "node-a"),
        (3.0, "node-b"),
        (4.0, "node-a"),
    ]
    assert trace.instances == ["node-a", "node-b"]
    assert trace.files == [a, b]
    assert trace.meta["cache_event_schema_version"] == CACHE_EVENT_SCHEMA_VERSION


def test_load_keeps_a_servers_order_at_equal_timestamps(tmp_path):
    path = _write(
        str(tmp_path / "a.lct"),
        [
            (5.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
            (5.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 2)),
            (5.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 3)),
        ],
    )

    trace = EventsTrace.load([path])

    assert [r.args["seq"] for r in trace.records] == [1, 2, 3]


def test_load_refuses_a_storage_level_file(tmp_path):
    path = str(tmp_path / "storage.lct")
    StorageTraceRecorder(path).close()

    with pytest.raises(ValueError, match="'storage'"):
        EventsTrace.load([path])


def test_load_refuses_another_cache_event_schema(tmp_path):
    path = _write(str(tmp_path / "future.lct"), [], schema=99)

    with pytest.raises(ValueError, match="99"):
        EventsTrace.load([path])


# -- Replay --------------------------------------------------------------------


def test_replay_delivers_in_order_and_counts(tmp_path):
    path = _write(
        str(tmp_path / "a.lct"),
        [
            (1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1)),
            (2.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
            (3.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 2)),
            (4.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 2)),
            (5.0, EVENTS_TRACE_BATCH, _batch("node-a", 2, 1)),
            (6.0, EVENTS_TRACE_LIFECYCLE, _stop("node-a")),
        ],
    )
    target = _RecordingTarget()

    result = asyncio.run(replay(EventsTrace.load([path]), target))

    assert target.calls == [
        ("start", EmitterIdentity("node-a", 1, "", 8000, 0)),
        ("batch", ("node-a", 1)),
        ("batch", ("node-a", 2)),
        ("start", EmitterIdentity("node-a", 2, "", 8000, 0)),
        ("batch", ("node-a", 1)),
        ("stop", "node-a"),
    ]
    assert (result.records, result.batches, result.starts, result.stops) == (
        6,
        3,
        2,
        1,
    )
    assert result.ingested == Ingested(applied=3)
    assert (result.ended, result.skipped) == (0, 0)


def test_replay_ends_a_stream_that_has_no_stop_mark(tmp_path):
    """node-a was killed (its file just ends); node-b shut down cleanly.
    The target hears ``end`` for the first only, once its last record went."""
    a = _write(
        str(tmp_path / "a.lct"),
        [
            (1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1)),
            (2.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
        ],
    )
    b = _write(
        str(tmp_path / "b.lct"),
        [
            (1.5, EVENTS_TRACE_LIFECYCLE, _start("node-b", 1)),
            (3.0, EVENTS_TRACE_BATCH, _batch("node-b", 1, 1)),
            (4.0, EVENTS_TRACE_LIFECYCLE, _stop("node-b")),
        ],
    )
    target = _RecordingTarget()

    result = asyncio.run(replay(EventsTrace.load([a, b]), target))

    assert [c for c in target.calls if c[0] in ("end", "stop")] == [
        ("end", "node-a"),
        ("stop", "node-b"),
    ]
    assert target.calls.index(("end", "node-a")) == 3
    assert (result.ended, result.stops) == (1, 1)


def test_replay_skips_what_it_cannot_deliver(tmp_path):
    """An unknown record kind, and a mark that names no instance."""
    path = _write(
        str(tmp_path / "a.lct"),
        [
            (1.0, "something.else", {"x": 1}),
            (2.0, EVENTS_TRACE_LIFECYCLE, {"phase": "stop"}),
        ],
    )
    target = _RecordingTarget()

    result = asyncio.run(replay(EventsTrace.load([path]), target))

    assert target.calls == []
    assert result.skipped == 2
    assert result.records == 2


def test_replay_paces_by_the_recorded_clock_when_asked(tmp_path):
    path = _write(
        str(tmp_path / "a.lct"),
        [
            (10.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
            (10.3, EVENTS_TRACE_BATCH, _batch("node-a", 1, 2)),
        ],
    )
    trace = EventsTrace.load([path])

    t0 = time.monotonic()
    asyncio.run(replay(trace, _RecordingTarget(), speed=3.0))
    paced = time.monotonic() - t0
    t0 = time.monotonic()
    asyncio.run(replay(trace, _RecordingTarget()))
    unpaced = time.monotonic() - t0

    assert paced >= 0.1
    assert unpaced < 0.1


def test_replay_rejects_a_negative_speed():
    with pytest.raises(ValueError, match="speed"):
        asyncio.run(replay(EventsTrace(), _RecordingTarget(), speed=-1.0))


# -- CoordinatorTarget against the real app ------------------------------------


def _app() -> FastAPI:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return create_app(config)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://coordinator"
    )


async def _instances(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    response = await client.get("/instances")
    response.raise_for_status()
    return response.json()["instances"]


async def _placements(client: httpx.AsyncClient, chunk: str) -> int:
    response = await client.post(
        "/directory/lookup",
        json={"keys": [_key(chunk).to_encoded_object_key().__dict__]},
    )
    response.raise_for_status()
    return len(response.json()["results"][0]["placements"])


def test_coordinator_target_registers_reports_and_deregisters(tmp_path):
    running = _write(
        str(tmp_path / "running.lct"),
        [
            (1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1, http_port=8123)),
            (2.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1, chunk="aa")),
            (3.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1, chunk="aa")),
        ],
    )
    stopping = _write(
        str(tmp_path / "stopping.lct"),
        [(4.0, EVENTS_TRACE_LIFECYCLE, _stop("node-a"))],
    )

    async def scenario() -> tuple:
        async with (
            _client(_app()) as client,
            CoordinatorTarget(client, "http://coordinator") as target,
        ):
            result = await replay(EventsTrace.load([running]), target)
            instances = await _instances(client)
            placements = await _placements(client, "aa")
            await replay(EventsTrace.load([stopping]), target)
            return result, instances, placements, await _instances(client)

    result, instances, placements, after_stop = asyncio.run(scenario())

    assert result.ingested == Ingested(applied=1, duplicates=1)
    assert [(i["instance_id"], i["ip"], i["http_port"]) for i in instances] == [
        ("node-a", "127.0.0.1", 8123)
    ]
    assert placements == 1
    assert after_stop == []


def test_coordinator_target_registers_at_the_recorded_address(tmp_path):
    mark = {**_start("node-a", 1, http_port=8123), "ip": "10.0.0.9"}
    path = _write(str(tmp_path / "a.lct"), [(1.0, EVENTS_TRACE_LIFECYCLE, mark)])

    async def scenario() -> list[dict[str, Any]]:
        async with (
            _client(_app()) as client,
            CoordinatorTarget(client, "http://coordinator") as target,
        ):
            await replay(EventsTrace.load([path]), target)
            return await _instances(client)

    assert [(i["ip"], i["http_port"]) for i in asyncio.run(scenario())] == [
        ("10.0.0.9", 8123)
    ]


def test_coordinator_target_heartbeats_until_the_stream_ends():
    """While a server's stream is live its heartbeat keeps advancing; once
    the stream ends without a stop, heartbeats cease and the coordinator's
    own stale sweep retires it, as it would a crashed server."""
    app = _app()
    registry = app.state.ctx.views.get(InstanceRegistry)

    async def scenario() -> tuple[float, float, float, list[str]]:
        async with (
            _client(app) as client,
            CoordinatorTarget(
                client, "http://coordinator", heartbeat_interval=0.02
            ) as target,
        ):
            # Straight to the target: through ``replay`` a one-record
            # stream would already have ended.
            await target.start(EmitterIdentity("node-a", 1, "", 8000, 0))
            first = registry.get("node-a").last_heartbeat_time
            await asyncio.sleep(0.1)
            second = registry.get("node-a").last_heartbeat_time
            await target.end("node-a")
            await asyncio.sleep(0.1)
            third = registry.get("node-a").last_heartbeat_time
            evicted = evict_stale(registry, instance_timeout=0.05)
            return first, second, third, evicted

    first, second, third, evicted = asyncio.run(scenario())

    assert second > first
    assert third == second
    assert evicted == ["node-a"]


def test_coordinator_target_ends_heartbeats_for_a_stream_without_a_stop(tmp_path):
    """Through ``replay`` itself: the file just ends, so the replayer calls
    ``end`` and the heartbeat task is gone before ``replay`` returns."""
    path = _write(
        str(tmp_path / "a.lct"),
        [
            (1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1)),
            (2.0, EVENTS_TRACE_BATCH, _batch("node-a", 1, 1)),
        ],
    )
    app = _app()
    registry = app.state.ctx.views.get(InstanceRegistry)

    async def scenario() -> tuple[int, float, float]:
        async with (
            _client(app) as client,
            CoordinatorTarget(
                client, "http://coordinator", heartbeat_interval=0.02
            ) as target,
        ):
            result = await replay(EventsTrace.load([path]), target)
            before = registry.get("node-a").last_heartbeat_time
            await asyncio.sleep(0.1)
            return result.ended, before, registry.get("node-a").last_heartbeat_time

    ended, before, after = asyncio.run(scenario())

    assert ended == 1
    assert after == before


def test_coordinator_target_rejects_a_negative_heartbeat_interval():
    with pytest.raises(ValueError, match="heartbeat_interval"):
        CoordinatorTarget(httpx.AsyncClient(), "http://c", heartbeat_interval=-1)


def test_coordinator_target_does_not_register_a_server_without_a_port(tmp_path):
    path = _write(
        str(tmp_path / "a.lct"),
        [(1.0, EVENTS_TRACE_LIFECYCLE, _start("node-a", 1, http_port=0))],
    )

    async def scenario() -> list[dict[str, Any]]:
        async with _client(_app()) as client:
            await replay(
                EventsTrace.load([path]),
                CoordinatorTarget(client, "http://coordinator"),
            )
            return await _instances(client)

    assert asyncio.run(scenario()) == []
