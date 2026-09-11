# SPDX-License-Identifier: Apache-2.0

"""Tests for TransferPhaseMetricsSubscriber (gather/DMA phase throughput)."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.metrics.phase_timing import (
    TransferPhaseMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_STAGING_METRIC = "lmcache_mp.transfer_staging_throughput"
_BYTES_METRIC = "lmcache_mp.transfer_phase_bytes"
_ELAPSED_METRIC = "lmcache_mp.transfer_phase_elapsed"


def _data_points(name: str) -> list:
    data = _reader.get_metrics_data()
    if data is None:
        return []
    return [
        dp
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for dp in metric.data.data_points
    ]


def _total_count(name: str) -> int:
    return sum(dp.count for dp in _data_points(name))


def _counter_total(name: str) -> float:
    return sum(dp.value for dp in _data_points(name))


def _attribute_sets(name: str) -> list[dict]:
    return [dict(dp.attributes) for dp in _data_points(name)]


def _handle(subscriber: TransferPhaseMetricsSubscriber, samples: list) -> None:
    handler = subscriber.get_subscriptions()[EventType.MP_TRANSFER_PHASE_SAMPLES]
    handler(
        Event(
            event_type=EventType.MP_TRANSFER_PHASE_SAMPLES,
            metadata={"samples": samples},
        )
    )


def test_only_staging_gets_a_throughput_histogram():
    """Both phases feed the byte/time counters; only staging is a throughput.

    A kernel section's elapsed is dominated by waiting for the co-resident
    inference engine to release the SMs, so a bytes/elapsed ratio for it would
    report contention rather than the kernel -- see the subscriber's docstring.
    """
    subscriber = TransferPhaseMetricsSubscriber()
    staging_before = _total_count(_STAGING_METRIC)
    bytes_before = _counter_total(_BYTES_METRIC)
    # (phase, direction, device_index, elapsed_ms, nbytes, session_id, t0, t1)
    _handle(
        subscriber,
        [(0, 1, 0, 100.0, 10**9, "s", 0.0, 0.0), (1, 1, 0, 50.0, 10**9, "s", 0.0, 0.0)],
    )
    # One staging sample in the histogram, and both phases' bytes in the
    # counter -- 1 GB each, so the counter advances by both samples' payload.
    assert _total_count(_STAGING_METRIC) == staging_before + 1
    assert _counter_total(_BYTES_METRIC) == bytes_before + 2 * 10**9


@pytest.mark.parametrize(
    "sample",
    [
        (0, 1, 0, 100.0),  # wrong arity
        (0, 1, 0, 0.0, 10**9, "s", 0.0, 0.0),  # non-positive time
        (0, 1, 0, 100.0, 0, "s", 0.0, 0.0),  # non-positive bytes
        (7, 1, 0, 100.0, 10**9, "s", 0.0, 0.0),  # unknown phase
        (0, 1, 0, "100.0", 10**9, "s", 0.0, 0.0),  # non-numeric time
        (0, 1, 0, 100.0, "big", "s", 0.0, 0.0),  # non-numeric bytes
        ("kernel", 1, 0, 100.0, 10**9, "s", 0.0, 0.0),  # non-numeric phase
    ],
)
def test_malformed_samples_dropped(sample):
    subscriber = TransferPhaseMetricsSubscriber()
    staging_before = _total_count(_STAGING_METRIC)
    bytes_before = _counter_total(_BYTES_METRIC)
    _handle(subscriber, [sample])
    assert _total_count(_STAGING_METRIC) == staging_before
    assert _counter_total(_BYTES_METRIC) == bytes_before


def test_phase_counters_accumulate_bytes_and_elapsed():
    subscriber = TransferPhaseMetricsSubscriber()
    bytes_before = _counter_total(_BYTES_METRIC)
    elapsed_before = _counter_total(_ELAPSED_METRIC)
    # 100 ms kernel + 400 ms staging over 1 GB + 3 GB.
    _handle(
        subscriber,
        [
            (0, 1, 0, 100.0, 10**9, "s", 0.0, 0.0),
            (1, 1, 0, 400.0, 3 * 10**9, "s", 0.0, 0.0),
        ],
    )
    assert _counter_total(_BYTES_METRIC) == bytes_before + 4 * 10**9
    assert _counter_total(_ELAPSED_METRIC) == pytest.approx(elapsed_before + 0.5)


def test_sample_labels():
    """Histograms carry device_index/direction; counters add the phase."""
    subscriber = TransferPhaseMetricsSubscriber()
    _handle(subscriber, [(1, 0, 3, 50.0, 10**9, "s", 0.0, 0.0)])
    assert {"device_index": "3", "direction": "h2d"} in _attribute_sets(_STAGING_METRIC)
    expected = {"device_index": "3", "direction": "h2d", "phase": "staging"}
    assert expected in _attribute_sets(_BYTES_METRIC)
    assert expected in _attribute_sets(_ELAPSED_METRIC)
