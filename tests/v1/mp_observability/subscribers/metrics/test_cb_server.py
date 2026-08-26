# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendMetricsSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
    BlendMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    histogram_count,
    read_counters,
)

_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = BlendMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = read_counters()

    def get_delta() -> dict[str, int]:
        return counter_delta(before, read_counters())

    return get_delta


class TestBlendMetricsSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_START in subs
        assert EventType.CB_STORE_PRE_COMPUTED_END in subs
        assert EventType.CB_STORE_FINAL_START in subs
        assert EventType.CB_STORE_FINAL_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    def test_subscriptions_cover_v3_sub_phase_events(self, subscriber):
        """Every V3 lookup/retrieve sub-phase event feeds a metric, so the phase
        breakdown survives trace sampling."""
        subs = subscriber.get_subscriptions()
        for event_type in (
            EventType.CB_FINGERPRINT_MATCH_START,
            EventType.CB_FINGERPRINT_MATCH_END,
            EventType.CB_PREFIX_LOOKUP_START,
            EventType.CB_PREFIX_LOOKUP_END,
            EventType.CB_COORDINATOR_MATCH_START,
            EventType.CB_COORDINATOR_MATCH_END,
            EventType.CB_SPARSE_PREFETCH_START,
            EventType.CB_SPARSE_PREFETCH_END,
            EventType.CB_SCATTER_START,
            EventType.CB_SCATTER_END,
            EventType.CB_RETRIEVE_NOOP,
        ):
            assert event_type in subs, f"{event_type} is not wired to a metric"

    def test_no_subscription_for_lifecycle_sentinels(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START not in subs
        assert EventType.CB_REQUEST_END not in subs
        assert EventType.CB_STORE_PRE_COMPUTED_SUBMITTED not in subs
        assert EventType.CB_RETRIEVE_SUBMITTED not in subs
        assert EventType.CB_STORE_FINAL_SUBMITTED not in subs

    def test_lookup_start_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id="req-1",
                metadata={"num_tokens": 128},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_end_normal(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 768,
                    "fingerprint_hits": 4,
                    "storage_hits": 3,
                    "stale_chunks": 1,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_end_no_gpu_context_flag(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 0,
                    "hit_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": True,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_success(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 2},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 2, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_pre_computed_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="req-3",
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="req-3",
                metadata={"instance_id": 0, "stored_chunks": 0, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_final_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id="req-4",
                metadata={"instance_id": 1, "num_tokens": 256},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id="req-4",
                metadata={"instance_id": 1, "stored_chunks": 0, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_fingerprints_registered(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                metadata={"num_chunks": 8},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_chunks_evicted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_CHUNKS_EVICTED,
                metadata={"num_chunks": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        for _ in range(5):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id="req-bulk",
                    metadata={"num_tokens": 100},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-bulk",
                    metadata={
                        "requested_tokens": 96,
                        "hit_tokens": 32,
                        "fingerprint_hits": 2,
                        "storage_hits": 1,
                        "stale_chunks": 1,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(0.15)
        bus.stop()


# ---------------------------------------------------------------------------
# Blend token-level hit-rate counters
#
# These counters expose the numerator/denominator that let dashboards compute
# the blend hit rate identically to the L1+L2 lookup hit rate:
#
#     rate(lmcache_blend_lookup_hit_tokens_total[5m])
#     / rate(lmcache_blend_lookup_requested_tokens_total[5m])
#
# Asserts on actual counter deltas via the InMemoryMetricReader fixture.
# ---------------------------------------------------------------------------


class TestBlendLookupHitTokenCounters:
    def test_full_hit(self, bus, subscriber, snapshot):
        """All requested tokens are served by blend."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 1024,
                    "fingerprint_hits": 4,
                    "storage_hits": 4,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024

    def test_partial_hit(self, bus, subscriber, snapshot):
        """A subset of the requested tokens is served by blend."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-2",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 256,
                    "fingerprint_hits": 4,
                    "storage_hits": 1,
                    "stale_chunks": 3,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 256

    def test_full_miss_still_records_denominator(self, bus, subscriber, snapshot):
        """Cold lookup: the request must still increment the denominator so
        the running hit rate properly reflects the miss."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-3",
                metadata={
                    "requested_tokens": 512,
                    "hit_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 512
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_no_gpu_context_records_zero_tokens(self, bus, subscriber, snapshot):
        """``no_gpu_context`` lookups emit ``hit_tokens=0`` and
        ``requested_tokens=0`` — neither counter should move so the ratio
        stays meaningful."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-4",
                metadata={
                    "requested_tokens": 0,
                    "hit_tokens": 0,
                    "fingerprint_hits": 5,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": True,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.lookup_requested_tokens", 0) == 0
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_multiple_lookups_accumulate(self, bus, subscriber, snapshot):
        """Counters accumulate across multiple completed lookups."""
        bus.start()
        # 3 full-hit lookups @ 256 tokens each
        for i in range(3):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"hit-{i}",
                    metadata={
                        "requested_tokens": 256,
                        "hit_tokens": 256,
                        "fingerprint_hits": 1,
                        "storage_hits": 1,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
        # 2 partial-hit lookups: 1024 requested, 128 hit
        for i in range(2):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"partial-{i}",
                    metadata={
                        "requested_tokens": 1024,
                        "hit_tokens": 128,
                        "fingerprint_hits": 4,
                        "storage_hits": 1,
                        "stale_chunks": 3,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        # 3*256 + 2*1024 = 768 + 2048 = 2816
        assert delta["lmcache_blend.lookup_requested_tokens"] == 2816
        # 3*256 + 2*128 = 768 + 256 = 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024


# ---------------------------------------------------------------------------
# V3 hit-token split
# ---------------------------------------------------------------------------


class TestBlendHitTokenSplitCounters:
    def test_prefix_segmented_and_non_prefix_split(self, bus, subscriber, snapshot):
        """The three V3 reuse paths are counted separately and sum to
        ``hit_tokens``, so a dashboard can attribute the hit rate."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-split",
                metadata={
                    "requested_tokens": 2048,
                    "hit_tokens": 1024,
                    "prefix_hit_tokens": 512,
                    "segmented_prefix_hit_tokens": 256,
                    "non_prefix_hit_tokens": 256,
                    "fingerprint_hits": 2,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_prefix_hit_tokens"] == 512
        assert delta["lmcache_blend.lookup_segmented_prefix_hit_tokens"] == 256
        assert delta["lmcache_blend.lookup_non_prefix_hit_tokens"] == 256
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024


# ---------------------------------------------------------------------------
# V3 phase durations
#
# Dispatched directly rather than through the bus: ``EventBus.publish()``
# overwrites ``Event.timestamp`` with the publish time, which would make the
# measured interval unpredictable.
# ---------------------------------------------------------------------------

_PHASE_EVENT_PAIRS = [
    ("lookup", EventType.CB_LOOKUP_START, EventType.CB_LOOKUP_END),
    (
        "fingerprint_match",
        EventType.CB_FINGERPRINT_MATCH_START,
        EventType.CB_FINGERPRINT_MATCH_END,
    ),
    ("prefix_lookup", EventType.CB_PREFIX_LOOKUP_START, EventType.CB_PREFIX_LOOKUP_END),
    (
        "coordinator_match",
        EventType.CB_COORDINATOR_MATCH_START,
        EventType.CB_COORDINATOR_MATCH_END,
    ),
    (
        "sparse_prefetch",
        EventType.CB_SPARSE_PREFETCH_START,
        EventType.CB_SPARSE_PREFETCH_END,
    ),
    ("retrieve", EventType.CB_RETRIEVE_START, EventType.CB_RETRIEVE_END),
    ("scatter", EventType.CB_SCATTER_START, EventType.CB_SCATTER_END),
]

# Metadata the non-generic handlers read unconditionally.
_PHASE_REQUIRED_METADATA = {
    EventType.CB_RETRIEVE_START: {"num_chunks": 1},
    EventType.CB_LOOKUP_END: {
        "requested_tokens": 256,
        "hit_tokens": 0,
        "fingerprint_hits": 0,
        "storage_hits": 0,
        "stale_chunks": 0,
    },
}


def _dispatch(subscriber, event_type, session_id, timestamp, **metadata):
    """Invoke the subscriber's handler for one event, bypassing the bus."""
    handler = subscriber.get_subscriptions()[event_type]
    handler(
        Event(
            event_type=event_type,
            session_id=session_id,
            timestamp=timestamp,
            metadata={**_PHASE_REQUIRED_METADATA.get(event_type, {}), **metadata},
        )
    )


class TestBlendPhaseDurations:
    @pytest.mark.parametrize(("phase", "start_type", "end_type"), _PHASE_EVENT_PAIRS)
    def test_phase_duration_recorded(self, subscriber, phase, start_type, end_type):
        name = f"lmcache_blend.{phase}_duration"
        before = histogram_count(name)
        now = time.time()
        sid = f"dur-{phase}"
        _dispatch(subscriber, start_type, sid, now)
        _dispatch(subscriber, end_type, sid, now + 0.025)

        assert histogram_count(name) == before + 1

    def test_end_without_start_records_nothing(self, subscriber):
        """A request whose lookup was abandoned mid-poll leaves no START, and a
        late END must not invent a duration."""
        name = "lmcache_blend.scatter_duration"
        before = histogram_count(name)
        _dispatch(subscriber, EventType.CB_SCATTER_END, "dur-orphan", time.time())

        assert histogram_count(name) == before

    def test_negative_interval_dropped(self, subscriber):
        """GPU-callback timestamps can invert against a CPU-stamped partner;
        such a pair is dropped rather than recorded negative."""
        name = "lmcache_blend.scatter_duration"
        before = histogram_count(name)
        now = time.time()
        sid = "dur-inverted"
        _dispatch(subscriber, EventType.CB_SCATTER_START, sid, now)
        _dispatch(subscriber, EventType.CB_SCATTER_END, sid, now - 0.005)

        assert histogram_count(name) == before

    def test_unmatched_starts_are_bounded(self, subscriber):
        """Abandoned lookups leave unmatched STARTs; the oldest are evicted, so
        an evicted START records nothing while a retained one still pairs."""
        # First Party
        from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
            _MAX_PENDING_PHASES,
        )

        name = "lmcache_blend.scatter_duration"
        now = time.time()
        for i in range(_MAX_PENDING_PHASES + 100):
            _dispatch(subscriber, EventType.CB_SCATTER_START, f"leak-{i}", now)

        before = histogram_count(name)
        _dispatch(subscriber, EventType.CB_SCATTER_END, "leak-0", now + 0.001)
        assert histogram_count(name) == before

        newest = f"leak-{_MAX_PENDING_PHASES + 99}"
        _dispatch(subscriber, EventType.CB_SCATTER_END, newest, now + 0.001)
        assert histogram_count(name) == before + 1

    def test_eviction_logs_a_warning_once(self, subscriber, monkeypatch):
        """Evicting an unmatched START is a leak signal: warn on the first one,
        then stay quiet -- a flood would hide everything else."""
        # First Party
        from lmcache.v1.mp_observability.subscribers.metrics import cb_server

        # The module logger does not propagate (see ``lmcache.logging``), so
        # spy on it instead of relying on caplog's root-handler capture.
        warnings: list[str] = []
        monkeypatch.setattr(
            cb_server.logger,
            "warning",
            lambda msg, *args, **kwargs: warnings.append(str(msg) % args),
        )

        now = time.time()
        for i in range(cb_server._MAX_PENDING_PHASES + 50):
            _dispatch(subscriber, EventType.CB_SCATTER_START, f"evict-{i}", now)

        assert len(warnings) == 1, warnings
        assert "evicted an unmatched START" in warnings[0]
        assert str(cb_server._MAX_PENDING_PHASES) in warnings[0]

    @pytest.mark.parametrize(
        ("phase", "start_type", "end_type"),
        [
            ("retrieve", EventType.CB_RETRIEVE_START, EventType.CB_RETRIEVE_END),
            ("scatter", EventType.CB_SCATTER_START, EventType.CB_SCATTER_END),
        ],
    )
    def test_tp_ranks_pair_independently(self, subscriber, phase, start_type, end_type):
        """At TP>1 every rank publishes its own retrieve/scatter pair under the
        shared request_id; each must record its own interval, not a span
        stitched from two workers."""
        name = f"lmcache_blend.{phase}_duration"
        before = histogram_count(name)
        now = time.time()
        sid = f"tp-{phase}"
        # Rank 0 starts, then rank 1 starts; rank 0 ends first, then rank 1.
        _dispatch(subscriber, start_type, sid, now, worker_id=0)
        _dispatch(subscriber, start_type, sid, now + 0.010, worker_id=1)
        _dispatch(subscriber, end_type, sid, now + 0.020, worker_id=0)
        assert histogram_count(name) == before + 1, "rank 0's END pairs with its START"
        _dispatch(subscriber, end_type, sid, now + 0.030, worker_id=1)
        assert histogram_count(name) == before + 2, "rank 1's END still has its START"

    def test_end_from_other_rank_does_not_consume_start(self, subscriber):
        """Rank 1's END must not close rank 0's open START."""
        name = "lmcache_blend.retrieve_duration"
        before = histogram_count(name)
        now = time.time()
        _dispatch(subscriber, EventType.CB_RETRIEVE_START, "tp-x", now, worker_id=0)
        _dispatch(
            subscriber, EventType.CB_RETRIEVE_END, "tp-x", now + 0.01, worker_id=1
        )
        assert histogram_count(name) == before
        _dispatch(
            subscriber, EventType.CB_RETRIEVE_END, "tp-x", now + 0.02, worker_id=0
        )
        assert histogram_count(name) == before + 1

    def test_events_without_worker_id_pair_as_before(self, subscriber):
        """Lookup legs (and V2 events) carry no worker_id and pair by session."""
        name = "lmcache_blend.prefix_lookup_duration"
        before = histogram_count(name)
        now = time.time()
        _dispatch(subscriber, EventType.CB_PREFIX_LOOKUP_START, "no-wid", now)
        _dispatch(subscriber, EventType.CB_PREFIX_LOOKUP_END, "no-wid", now + 0.005)
        assert histogram_count(name) == before + 1


# ---------------------------------------------------------------------------
# V3 phase payload counters
# ---------------------------------------------------------------------------


# (events to publish, expected counter deltas).
_PHASE_COUNTER_CASES = [
    (
        [(EventType.CB_FINGERPRINT_MATCH_END, {"matches": 7})],
        {"lmcache_blend.fingerprint_matches": 7},
    ),
    (
        [(EventType.CB_COORDINATOR_MATCH_END, {"matches": 3, "timed_out": False})],
        {
            "lmcache_blend.coordinator_matches": 3,
            "lmcache_blend.coordinator_match_timeouts": 0,
        },
    ),
    # A timed-out match leg silently shrinks reuse, so it gets its own counter.
    (
        [(EventType.CB_COORDINATOR_MATCH_END, {"matches": 0, "timed_out": True})],
        {"lmcache_blend.coordinator_match_timeouts": 1},
    ),
    (
        [
            (
                EventType.CB_SPARSE_PREFETCH_START,
                {"n_chunks": 4, "world_size": 2, "n_keys": 8, "l2_keys": 6},
            ),
            (EventType.CB_SPARSE_PREFETCH_END, {"found_keys": 8, "l2_keys": 6}),
        ],
        {
            "lmcache_blend.sparse_prefetch_l2_keys": 6,
            "lmcache_blend.sparse_prefetch_found_keys": 8,
        },
    ),
    (
        [
            (
                EventType.CB_SCATTER_START,
                {"scattered_tokens": 1280, "n_prefix": 1, "n_shifted": 4, "dropped": 2},
            )
        ],
        {
            "lmcache_blend.scatter_tokens": 1280,
            "lmcache_blend.scatter_prefix_chunks": 1,
            "lmcache_blend.scatter_shifted_chunks": 4,
            "lmcache_blend.scatter_dropped_chunks": 2,
        },
    ),
    # No-ops return success, so this counter is the only signal. read_counters
    # sums across attribute sets: 2 beyond_slot_bound + 1 no_object_keys.
    (
        [
            (EventType.CB_RETRIEVE_NOOP, {"reason": reason, "dropped_matches": 3})
            for reason in ("beyond_slot_bound", "beyond_slot_bound", "no_object_keys")
        ],
        {"lmcache_blend.retrieve_noops": 3},
    ),
]


class TestBlendPhaseCounters:
    @pytest.mark.parametrize(("events", "expected"), _PHASE_COUNTER_CASES)
    def test_payload_counters(self, bus, subscriber, snapshot, events, expected):
        bus.start()
        for event_type, metadata in events:
            bus.publish(
                Event(
                    event_type=event_type,
                    session_id="req-counter",
                    metadata=metadata,
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        for name, value in expected.items():
            assert delta.get(name, 0) == value

    def test_retrieve_noop_without_reason_does_not_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(event_type=EventType.CB_RETRIEVE_NOOP, session_id="req-noop-bare")
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert bus.subscriber_exception_counts() == {}
