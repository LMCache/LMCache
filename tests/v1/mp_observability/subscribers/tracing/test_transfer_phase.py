# SPDX-License-Identifier: Apache-2.0
"""Tests for TransferPhaseTracingSubscriber."""

# Standard
from unittest.mock import patch
import time

# Third Party
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing import (
    mp_server as mp_server_module,
)
from lmcache.v1.mp_observability.subscribers.tracing import (
    transfer_phase as transfer_phase_module,
)
from lmcache.v1.mp_observability.subscribers.tracing.mp_server import (
    MPServerTracingSubscriber,
)
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry
from lmcache.v1.mp_observability.subscribers.tracing.transfer_phase import (
    TransferPhaseTracingSubscriber,
)
from lmcache.v1.platform.ops_types import TransferPhase

D2H = int(TransferDirection.D2H)
H2D = int(TransferDirection.H2D)
KERNEL = int(TransferPhase.KERNEL)
STAGING = int(TransferPhase.STAGING)
MB = 1 << 20


@pytest.fixture
def exporter():
    """Real OTel provider with an in-memory exporter, patched into both
    tracing modules so parent and child spans share one tracer."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracer = provider.get_tracer("lmcache_mp.test")
    with (
        patch.object(mp_server_module, "_tracer", tracer),
        patch.object(mp_server_module, "_HAS_OTEL", True),
        patch.object(transfer_phase_module, "_tracer", tracer),
        patch.object(transfer_phase_module, "_HAS_OTEL", True),
    ):
        yield exp
    exp.shutdown()


def _run(
    events: list[Event],
    parent_ttl_s: float = 120.0,
    settle_s: float = 0.0,
) -> None:
    """Dispatch *events* through a bus with both tracing subscribers.

    ``settle_s`` waits between publishing and stopping the bus.
    """
    bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
    registry = SpanRegistry()
    bus.register_subscriber(MPServerTracingSubscriber(registry))
    bus.register_subscriber(TransferPhaseTracingSubscriber(registry, parent_ttl_s))
    bus.start()
    for event in events:
        bus.publish(event)
        if settle_s:
            time.sleep(settle_s)
    time.sleep(0.15)
    bus.stop()


def _store_transfer_events(
    sid: str, now: float, offset: float, tkey: str
) -> list[Event]:
    """One store transfer's START/END pair, `offset` seconds into the request.

    `tkey` is the transfer key the phase-timing samples echo back; the
    subscriber matches on it, so the sample tuples must carry the same value
    in their session_id slot.
    """
    return [
        Event(
            event_type=EventType.MP_STORE_START,
            session_id=sid,
            timestamp=now + offset,
            metadata={
                "device": "cuda:0",
                "engine_id": 0,
                "model_name": "m",
                "transfer_key": tkey,
            },
        ),
        Event(
            event_type=EventType.MP_STORE_END,
            session_id=sid,
            timestamp=now + offset + 0.009,
            metadata={
                "device": "cuda:0",
                "stored_count": 1,
                "engine_id": 0,
                "model_name": "m",
                "cache_salt": "",
                "total_bytes": 0,
                "num_tokens": 0,
                "transfer_key": tkey,
            },
        ),
    ]


def _store_events(sid: str, now: float, tkey: str | None = None) -> list[Event]:
    return (
        [Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)]
        + _store_transfer_events(sid, now, 0.001, tkey or sid)
        + [
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.02,
            )
        ]
    )


def _samples_event(samples: list, now: float, sid: str = "other") -> Event:
    return Event(
        event_type=EventType.MP_TRANSFER_PHASE_SAMPLES,
        session_id=sid,
        timestamp=now,
        metadata={"samples": samples},
    )


def _spans_named(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_phases_stack_under_the_store_span(exporter):
    """Two batch steps -> one kernel span and one staging span, stacked:
    kernel (ran first) starts at the transfer's real start and lasts its busy
    time, staging follows for its elapsed; totals summed; nested under
    mp.store even though that span has already ended."""
    now = time.time()
    sid = "req-1"
    samples = [
        (KERNEL, D2H, 0, 1.0, MB, sid, now + 0.002, now + 0.003),
        (STAGING, D2H, 0, 2.0, MB, sid, now + 0.003, now + 0.005),
        (KERNEL, D2H, 0, 1.0, MB, sid, now + 0.005, now + 0.006),
        (STAGING, D2H, 0, 2.0, MB, sid, now + 0.006, now + 0.008),
    ]
    _run(_store_events(sid, now) + [_samples_event(samples, now + 0.5)])

    store = _spans_named(exporter, "mp.store")
    assert len(store) == 1
    kernel = _spans_named(exporter, "transfer.kernel_interval")
    staging = _spans_named(exporter, "transfer.staging")
    assert len(kernel) == 1 and len(staging) == 1
    for child in (kernel[0], staging[0]):
        assert child.parent is not None
        assert child.parent.span_id == store[0].context.span_id
        assert child.attributes["session_id"] == sid
        assert child.attributes["num_steps"] == 2
        assert child.attributes["nbytes"] == 2 * MB
    # Stacked: kernel = [start, start + 2 ms], staging = next 4 ms.
    assert kernel[0].start_time == int((now + 0.002) * 1e9)
    assert kernel[0].end_time - kernel[0].start_time == pytest.approx(2e6, abs=1e3)
    assert staging[0].start_time == kernel[0].end_time
    assert staging[0].end_time - staging[0].start_time == pytest.approx(4e6, abs=1e3)
    # Real intervals kept as attributes.
    assert kernel[0].attributes["first_start_s"] == pytest.approx(now + 0.002)
    assert kernel[0].attributes["last_end_s"] == pytest.approx(now + 0.006)
    assert staging[0].attributes["last_end_s"] == pytest.approx(now + 0.008)
    assert kernel[0].attributes["elapsed_seconds"] == pytest.approx(0.002)
    # Only staging carries a rate. A kernel section's elapsed is mostly the
    # wait for the co-resident engine's SMs, so nbytes/elapsed there would
    # report contention as if it were transfer rate.
    assert staging[0].attributes["throughput_GB_per_second"] == pytest.approx(
        2 * MB / 0.004 / 1e9
    )
    assert "throughput_GB_per_second" not in kernel[0].attributes


def test_samples_split_across_pops_merge_into_one_span(exporter):
    """Samples popped partly before and partly after MP_STORE_END fold into
    one span, emitted at the first samples event after END."""
    now = time.time()
    sid = "req-2"
    first = [(KERNEL, D2H, 0, 1.0, MB, sid, now + 0.002, now + 0.003)]
    second = [(KERNEL, D2H, 0, 1.0, MB, sid, now + 0.005, now + 0.006)]
    start, store_start, store_end, req_end = _store_events(sid, now)
    _run(
        [
            start,
            store_start,
            _samples_event(first, now + 0.004),  # popped mid-transfer
            store_end,
            req_end,
            _samples_event(second, now + 0.5),  # popped after END: the rest
        ]
    )
    kernel = _spans_named(exporter, "transfer.kernel_interval")
    assert len(kernel) == 1
    assert kernel[0].attributes["num_steps"] == 2
    assert kernel[0].attributes["first_start_s"] == pytest.approx(now + 0.002)
    assert kernel[0].attributes["last_end_s"] == pytest.approx(now + 0.006)


def test_not_emitted_before_end(exporter):
    """Without MP_STORE_END nothing is emitted until shutdown flushes the
    remainder."""
    now = time.time()
    sid = "req-2b"
    samples = [(KERNEL, D2H, 0, 1.0, MB, sid, now + 0.002, now + 0.003)]
    start, store_start, _, _ = _store_events(sid, now)
    bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
    registry = SpanRegistry()
    bus.register_subscriber(MPServerTracingSubscriber(registry))
    bus.register_subscriber(TransferPhaseTracingSubscriber(registry))
    bus.start()
    for event in (start, store_start, _samples_event(samples, now + 0.5)):
        bus.publish(event)
    time.sleep(0.15)
    assert _spans_named(exporter, "transfer.kernel_interval") == []
    bus.stop()  # shutdown flushes best-effort
    assert len(_spans_named(exporter, "transfer.kernel_interval")) == 1


def test_samples_without_wall_clock_emit_no_spans(exporter):
    """Samples whose anchor failed (bounds of 0.0) are ignored."""
    now = time.time()
    sid = "req-3"
    samples = [(KERNEL, D2H, 0, 1.0, MB, sid, 0.0, 0.0)]
    _run(_store_events(sid, now) + [_samples_event(samples, now + 0.5)])
    assert _spans_named(exporter, "transfer.kernel_interval") == []


def test_unknown_session_or_direction_mismatch_is_skipped(exporter):
    """No parent span for the session, or a retrieve sample against a store
    parent, produces no child span."""
    now = time.time()
    sid = "req-4"
    samples = [
        (KERNEL, D2H, 0, 1.0, MB, "never-seen", now, now + 0.001),
        (KERNEL, H2D, 0, 1.0, MB, sid, now, now + 0.001),
    ]
    _run(_store_events(sid, now) + [_samples_event(samples, now + 0.5)])
    assert _spans_named(exporter, "transfer.kernel_interval") == []


@pytest.mark.parametrize(
    "sample",
    [
        (KERNEL, D2H, 0, 1.0, MB),  # legacy arity
        (KERNEL, D2H, 0, 1.0, MB, "req-5", "x", "y"),  # non-numeric bounds
        (KERNEL, D2H, 0, 1.0, MB, "req-5", 5.0, 4.0),  # end before start
        (KERNEL, D2H, 0, "1", MB, "req-5", 1.0, 2.0),  # non-numeric elapsed
        (99, D2H, 0, 1.0, MB, "req-5", 1.0, 2.0),  # unknown phase
    ],
)
def test_malformed_samples_dropped(exporter, sample):
    now = time.time()
    _run(_store_events("req-5", now) + [_samples_event([sample], now + 0.5)])
    assert _spans_named(exporter, "transfer.kernel_interval") == []
    assert _spans_named(exporter, "transfer.staging") == []


def test_parent_expires_after_ttl(exporter):
    """A parent context older than the TTL is dropped before attachment."""
    now = time.time()
    sid = "req-6"
    samples = [(KERNEL, D2H, 0, 1.0, MB, sid, now, now + 0.001)]
    _run(
        _store_events(sid, now) + [_samples_event(samples, now + 0.5)],
        parent_ttl_s=0.05,
        settle_s=0.03,
    )
    assert _spans_named(exporter, "transfer.kernel_interval") == []


def test_second_transfer_of_a_session_does_not_reuse_a_released_parent(exporter):
    """Once a transfer's spans are emitted its parent context is released:
    samples for the same session arriving later find no parent."""
    now = time.time()
    sid = "req-7"
    first = [(KERNEL, D2H, 0, 1.0, MB, sid, now + 0.002, now + 0.003)]
    late = [(KERNEL, D2H, 0, 1.0, MB, sid, now + 0.005, now + 0.006)]
    _run(
        _store_events(sid, now)
        + [_samples_event(first, now + 0.5), _samples_event(late, now + 0.6)]
    )
    kernel = _spans_named(exporter, "transfer.kernel_interval")
    assert len(kernel) == 1
    assert kernel[0].attributes["num_steps"] == 1


def _two_transfer_case(exporter, order: str):
    """One request, two store transfers, samples arriving in `order`.

    `order="samples_first"` interleaves SAMPLES1 before START2 (the drain
    thread keeps up); `order="ends_first"` dispatches both ENDs before any
    samples, which is what happens whenever the drain thread lags -- the
    sampler publishes SAMPLES from inside an END handler, so they queue
    behind an already-published START/END of the next transfer.
    """
    now = time.time()
    sid = "req-two"
    k1, k2 = f"{sid}#1", f"{sid}#2"
    first = [
        (KERNEL, D2H, 0, 1.0, MB, k1, now + 0.002, now + 0.003),
        (STAGING, D2H, 0, 2.0, MB, k1, now + 0.003, now + 0.005),
    ]
    second = [
        (KERNEL, D2H, 0, 9.0, 2 * MB, k2, now + 0.012, now + 0.021),
        (STAGING, D2H, 0, 3.0, 2 * MB, k2, now + 0.021, now + 0.024),
    ]
    start1, end1 = _store_transfer_events(sid, now, 0.001, k1)
    start2, end2 = _store_transfer_events(sid, now, 0.011, k2)
    s1 = _samples_event(first, now + 0.5, sid)
    s2 = _samples_event(second, now + 0.6, sid)
    seq = (
        [start1, end1, s1, start2, end2, s2]
        if order == "samples_first"
        else [start1, end1, start2, end2, s1, s2]
    )
    _run(
        [Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)]
        + seq
        + [
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.04,
            )
        ]
    )
    store = _spans_named(exporter, "mp.store")
    kernel = _spans_named(exporter, "transfer.kernel_interval")
    staging = _spans_named(exporter, "transfer.staging")
    assert len(store) == 2
    assert len(kernel) == 2 and len(staging) == 2, (
        f"lost a transfer's phase spans ({order}): {len(kernel)}"
    )
    store_ids = {sp.context.span_id for sp in store}
    for child in kernel + staging:
        assert child.parent is not None and child.parent.span_id in store_ids
    # Each transfer keeps its own totals -- no folding into one.
    assert len({c.parent.span_id for c in kernel}) == 2
    assert sorted(c.attributes["nbytes"] for c in kernel) == [MB, 2 * MB]
    assert sorted(c.attributes["nbytes"] for c in staging) == [MB, 2 * MB]


def test_two_transfers_samples_between_ends(exporter):
    """Drain keeps up: SAMPLES1 lands before the next transfer starts."""
    _two_transfer_case(exporter, "samples_first")


def test_two_transfers_both_ends_before_samples(exporter):
    """Drain lags: both ENDs dispatch before either samples event.

    Order-based matching folds everything into the first transfer here and
    drops the second's spans; matching on the transfer key does not.
    """
    _two_transfer_case(exporter, "ends_first")
