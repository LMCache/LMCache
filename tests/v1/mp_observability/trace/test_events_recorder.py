# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``events`` trace level: recorder, sink, lifecycle, CLI.

Everything runs in-process on CPU: a ``CacheEventSubscriber`` fed by hand,
a ``TraceCacheEventSink`` writing into an ``EventsTraceRecorder``, and a
``TraceReader`` reading the file back.
"""

# Standard
from dataclasses import dataclass
import argparse
import contextlib
import io
import re

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace.info_command import InfoCommand
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CACHE_EVENT_SCHEMA_VERSION,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.cache_events import (
    EVENTS_TRACE_BATCH,
    EVENTS_TRACE_LIFECYCLE,
    CacheEventPublishError,
    CacheEventSink,
    MultiCacheEventSink,
    TraceCacheEventSink,
    TraceLifecyclePhase,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace import lifecycle
from lmcache.v1.mp_observability.trace.decorator import (
    is_tracing_enabled,
    set_tracing_enabled,
)
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import (
    EventsTraceRecorder,
    StorageTraceRecorder,
)


@pytest.fixture(autouse=True)
def _reset_gate_and_active_recorder():
    yield
    set_tracing_enabled(False)
    lifecycle._active_recorder = None  # noqa: SLF001 - test isolation of module state


@pytest.fixture
def trace_path(tmp_path):
    return str(tmp_path / "events.lct")


@dataclass
class _FakeStorageManagerCfg:
    chunk_size: int = 256
    adapters: tuple = ()


def _key(salt: str, chunk: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk), model_name="m", kv_rank=0, cache_salt=salt
    )


def _batch(
    seq: int,
    event_type: CacheEventType = CacheEventType.STORE,
    tier: Tier = Tier.L2,
    backend: str = "fs",
    shared: bool = False,
    size: int = 100,
    tokens: list[int] | None = None,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=7,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        shared=shared,
        ts=1.5,
        entries=[
            CacheEventEntry(
                key=_key("alice", "aa").to_encoded_object_key(),
                size_bytes=size if event_type is CacheEventType.STORE else 0,
                token_ids=tokens or [],
                token_offset=0 if tokens else -1,
            )
        ],
    )


def _records(path: str) -> list:
    with TraceReader(path) as reader:
        return list(reader.records())


# -- Recorder -----------------------------------------------------------------


class TestEventsTraceRecorder:
    def test_header_carries_the_level_and_its_metadata(self, trace_path):
        recorder = EventsTraceRecorder(
            trace_path, level_meta={"instance_id": "node-a", "x": 1}
        )
        recorder.attach_storage_config(_FakeStorageManagerCfg())
        recorder.close()

        with TraceReader(trace_path) as reader:
            header = reader.header
        assert header.level == "events"
        assert header.level_meta == {"instance_id": "node-a", "x": 1}
        assert header.sm_config_digest

    def test_leaves_the_call_tracing_gate_alone(self, trace_path):
        # No TRACE_CALL consumer exists at this level, so decorated
        # StorageManager calls must not start publishing for nobody.
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        assert is_tracing_enabled() is False
        recorder.close()
        assert is_tracing_enabled() is False

    def test_storage_recorder_still_flips_the_gate(self, tmp_path):
        recorder = StorageTraceRecorder(str(tmp_path / "s.lct"))
        assert is_tracing_enabled() is True
        recorder.close()
        assert is_tracing_enabled() is False

    def test_subscribes_to_nothing_but_closes_with_the_bus(self, trace_path):
        bus = EventBus(EventBusConfig())
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        assert recorder.get_subscriptions() == {}
        bus.register_subscriber(recorder)
        bus.stop()
        # A closed file is readable and has its header.
        with TraceReader(trace_path) as reader:
            assert reader.header.level == "events"

    def test_storage_header_has_empty_level_meta(self, tmp_path):
        path = str(tmp_path / "s.lct")
        StorageTraceRecorder(path).close()
        with TraceReader(path) as reader:
            assert reader.header.level_meta == {}


# -- Sink ---------------------------------------------------------------------


class TestTraceCacheEventSink:
    def test_records_batches_in_wire_form(self, trace_path):
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        sink = TraceCacheEventSink(recorder)
        batches = [
            _batch(1, tokens=[1, 2, 3]),
            _batch(2, event_type=CacheEventType.DELETE, tier=Tier.L1, backend="dram"),
        ]

        sink.publish(batches)
        recorder.close()

        records = _records(trace_path)
        assert [r.qualname for r in records] == [EVENTS_TRACE_BATCH] * 2
        # The stored form is exactly what POST /events would carry, so it
        # parses back through the request schema into the same batches.
        wire = CacheEventsRequest(batches=batches).model_dump(mode="json")
        assert [r.args for r in records] == wire["batches"]
        parsed = CacheEventsRequest.model_validate(
            {"batches": [r.args for r in records]}
        )
        assert parsed.batches == batches

    def test_lifecycle_marks_start_identity_and_stop(self, trace_path):
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        sink = TraceCacheEventSink(recorder)

        sink.record_lifecycle(
            TraceLifecyclePhase.START,
            instance_id="node-a",
            incarnation=7,
            ip="10.0.0.1",
            http_port=8000,
            mq_port=0,
        )
        sink.publish([_batch(1)])
        sink.close()
        recorder.close()

        records = _records(trace_path)
        assert [r.qualname for r in records] == [
            EVENTS_TRACE_LIFECYCLE,
            EVENTS_TRACE_BATCH,
            EVENTS_TRACE_LIFECYCLE,
        ]
        assert records[0].args == {
            "phase": "start",
            "instance_id": "node-a",
            "incarnation": 7,
            "ip": "10.0.0.1",
            "http_port": 8000,
            "mq_port": 0,
        }
        assert records[2].args == {"phase": "stop", "instance_id": "node-a"}

    def test_records_are_ordered_and_timed(self, trace_path):
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        sink = TraceCacheEventSink(recorder)
        sink.publish([_batch(1)])
        sink.publish([_batch(2)])
        recorder.close()

        records = _records(trace_path)
        assert [r.args["seq"] for r in records] == [1, 2]
        assert records[0].t_mono <= records[1].t_mono
        assert all(r.t_wall > 0 for r in records)

    def test_a_closed_recorder_counts_drops_instead_of_raising(self, trace_path):
        recorder = EventsTraceRecorder(trace_path, level_meta={})
        recorder.close()
        sink = TraceCacheEventSink(recorder)
        sink.publish([_batch(1)])  # must not raise on the drain thread
        assert recorder.dropped_count == 1


class _RecordingSink(CacheEventSink):
    def __init__(self, fail: bool = False) -> None:
        self.published: list[list[CacheEventBatch]] = []
        self.fail = fail
        self.closed = False

    def publish(self, batches: list[CacheEventBatch]) -> None:
        if self.fail:
            raise CacheEventPublishError("injected")
        self.published.append(batches)

    def close(self) -> None:
        self.closed = True


class TestMultiCacheEventSink:
    def test_delivers_to_every_sink_in_order(self):
        a, b = _RecordingSink(), _RecordingSink()
        MultiCacheEventSink([a, b]).publish([_batch(1)])
        assert len(a.published) == 1 and len(b.published) == 1

    def test_a_failing_sink_does_not_stop_the_others(self):
        a, b = _RecordingSink(fail=True), _RecordingSink()
        with pytest.raises(CacheEventPublishError, match="1 of 2"):
            MultiCacheEventSink([a, b]).publish([_batch(1)])
        assert len(b.published) == 1

    def test_close_closes_all(self):
        a, b = _RecordingSink(), _RecordingSink()
        MultiCacheEventSink([a, b]).close()
        assert a.closed and b.closed

    def test_needs_at_least_one_sink(self):
        with pytest.raises(ValueError):
            MultiCacheEventSink([])


# -- Lifecycle ----------------------------------------------------------------


class TestLifecycle:
    def test_events_level_builds_an_events_recorder_with_metadata(self, trace_path):
        bus = EventBus(EventBusConfig())
        cfg = ObservabilityConfig(trace_level="events", trace_output=trace_path)

        recorder = lifecycle.maybe_initialize_trace_recorder(
            bus, cfg, _FakeStorageManagerCfg(), instance_id="node-a"
        )

        assert isinstance(recorder, EventsTraceRecorder)
        assert lifecycle.get_active_trace_recorder() is recorder
        bus.stop()
        with TraceReader(trace_path) as reader:
            meta = reader.header.level_meta
        assert meta["instance_id"] == "node-a"
        assert meta["cache_event_schema_version"] == CACHE_EVENT_SCHEMA_VERSION
        assert meta["lmcache_version"]

    def test_storage_level_still_builds_a_storage_recorder(self, tmp_path):
        bus = EventBus(EventBusConfig())
        cfg = ObservabilityConfig(
            trace_level="storage", trace_output=str(tmp_path / "s.lct")
        )
        recorder = lifecycle.maybe_initialize_trace_recorder(
            bus, cfg, _FakeStorageManagerCfg()
        )
        assert isinstance(recorder, StorageTraceRecorder)
        bus.stop()

    def test_unknown_level_is_refused(self, trace_path):
        bus = EventBus(EventBusConfig())
        cfg = ObservabilityConfig(trace_level="gpu", trace_output=trace_path)
        with pytest.raises(ValueError, match="unsupported trace level"):
            lifecycle.maybe_initialize_trace_recorder(
                bus, cfg, _FakeStorageManagerCfg()
            )

    def test_no_level_means_no_recorder(self):
        bus = EventBus(EventBusConfig())
        assert (
            lifecycle.maybe_initialize_trace_recorder(
                bus, ObservabilityConfig(), _FakeStorageManagerCfg()
            )
            is None
        )
        assert lifecycle.get_active_trace_recorder() is None


# -- CLI ----------------------------------------------------------------------


def test_info_summarizes_an_events_trace(trace_path):
    recorder = EventsTraceRecorder(
        trace_path,
        level_meta={"instance_id": "node-a", "cache_event_schema_version": 1},
    )
    sink = TraceCacheEventSink(recorder)
    sink.record_lifecycle(
        TraceLifecyclePhase.START, instance_id="node-a", incarnation=7
    )
    sink.publish([_batch(1, size=100, tokens=[1, 2]), _batch(2, size=50)])
    sink.publish(
        [_batch(3, event_type=CacheEventType.DELETE, tier=Tier.L1, backend="dram")]
    )
    sink.close()
    recorder.close()

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        InfoCommand().execute(argparse.Namespace(trace_path=trace_path))
    text = out.getvalue()

    def row(label: str) -> str:
        match = re.search(rf"^\s*{label}:\s+(.*)$", text, re.MULTILINE)
        assert match, f"no {label!r} row in:\n{text}"
        return match.group(1)

    assert row("level") == "events"
    assert row("instance_id") == "node-a"
    assert "store/l2/fs/local: 2" in text
    assert "delete/l1/dram/local: 1" in text
    assert row("store_bytes") == "150"
    assert row("entries_with_tokens") == "1"
    assert row("lifecycle") == "start=1, stop=1"
