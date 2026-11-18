# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache trace replay-events``."""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio

# Third Party
import httpx
import pytest

# First Party
from lmcache.cli.commands.trace.replay_events_command import (
    ReplayEventsCommand,
    run_events_replay,
)
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CACHE_EVENT_SCHEMA_VERSION,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import (
    EVENTS_TRACE_BATCH,
    EVENTS_TRACE_LIFECYCLE,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.recorder import (
    EventsTraceRecorder,
    StorageTraceRecorder,
)


@pytest.fixture(autouse=True)
def _reset_trace_gate():
    yield
    set_tracing_enabled(False)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ReplayEventsCommand().add_arguments(parser)
    args = parser.parse_args(argv)
    args.quiet = True
    return args


def _events_file(path: str) -> str:
    key = ObjectKey(
        chunk_hash=bytes.fromhex("aa"), model_name="m", kv_rank=0, cache_salt="alice"
    )
    batch = CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=10)],
    )
    wire = CacheEventsRequest(batches=[batch]).model_dump(mode="json")["batches"][0]
    recorder = EventsTraceRecorder(
        path,
        level_meta={
            "instance_id": "node-a",
            "cache_event_schema_version": CACHE_EVENT_SCHEMA_VERSION,
        },
    )
    recorder.write_record(
        EVENTS_TRACE_LIFECYCLE,
        {
            "phase": "start",
            "instance_id": "node-a",
            "incarnation": 1,
            "ip": "",
            "http_port": 8000,
            "mq_port": 0,
        },
        t_wall=1.0,
        t_mono=0.0,
    )
    recorder.write_record(EVENTS_TRACE_BATCH, wire, t_wall=2.0, t_mono=1.0)
    recorder.close()
    return path


def test_parser_takes_several_files_and_the_replay_flags():
    args = _parse(
        [
            "a.lct",
            "b.lct",
            "--coordinator-url",
            "http://coordinator:9300",
            "--speed",
            "2",
            "--heartbeat-interval",
            "1.5",
        ]
    )

    assert args.trace_path == ["a.lct", "b.lct"]
    assert args.coordinator_url == "http://coordinator:9300"
    assert args.speed == 2.0
    assert args.heartbeat_interval == 1.5


def test_flags_default_to_an_unpaced_replay_with_heartbeats():
    args = _parse(["a.lct", "--coordinator-url", "http://c"])

    assert args.speed == 0.0
    assert args.heartbeat_interval == 5.0


def test_the_coordinator_url_is_required():
    with pytest.raises(SystemExit) as exit_info:
        _parse(["a.lct"])

    assert exit_info.value.code == 2


def test_a_storage_trace_is_refused(tmp_path):
    path = str(tmp_path / "storage.lct")
    StorageTraceRecorder(path).close()

    with pytest.raises(SystemExit) as exit_info:
        run_events_replay(_parse([path, "--coordinator-url", "http://c"]))

    assert exit_info.value.code == 2


def test_an_unreachable_coordinator_stops_the_replay_with_a_clean_exit(tmp_path):
    """Nothing listens on port 1: the first call fails, and the command ends
    with a message and status 1 rather than a traceback."""
    path = _events_file(str(tmp_path / "events.lct"))

    with pytest.raises(SystemExit) as exit_info:
        run_events_replay(_parse([path, "--coordinator-url", "http://127.0.0.1:1"]))

    assert exit_info.value.code == 1


def test_the_stream_is_delivered_to_the_coordinator(tmp_path, monkeypatch):
    """End to end through the command, against the real app in-process."""
    path = _events_file(str(tmp_path / "events.lct"))
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    app = create_app(config)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_kw: real_client(transport=httpx.ASGITransport(app=app)),
    )

    run_events_replay(_parse([path, "--coordinator-url", "http://coordinator"]))

    async def inspect() -> tuple[list, int]:
        async with real_client(
            transport=httpx.ASGITransport(app=app), base_url="http://coordinator"
        ) as client:
            instances = (await client.get("/instances")).json()["instances"]
            stats = (await client.get("/directory/stats")).json()
            return instances, stats["num_placements"]

    instances, placements = asyncio.run(inspect())
    assert [i["instance_id"] for i in instances] == ["node-a"]
    assert placements == 1
