# SPDX-License-Identifier: Apache-2.0

"""Tests for TransferPhaseSubscriber (gather/DMA phase throughput)."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.metrics.phase_timing import (
    TransferPhaseSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_KERNEL_METRIC = "lmcache_mp.transfer_kernel_throughput"
_STAGING_METRIC = "lmcache_mp.transfer_staging_throughput"


def _total_count(name: str) -> int:
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    return sum(
        dp.count
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for dp in metric.data.data_points
    )


def _handle(subscriber: TransferPhaseSubscriber, samples: list) -> None:
    handler = subscriber.get_subscriptions()[EventType.MP_TRANSFER_PHASE_SAMPLES]
    handler(
        Event(
            event_type=EventType.MP_TRANSFER_PHASE_SAMPLES,
            metadata={"samples": samples},
        )
    )


def test_records_both_phases():
    subscriber = TransferPhaseSubscriber()
    kernel_before = _total_count(_KERNEL_METRIC)
    staging_before = _total_count(_STAGING_METRIC)
    # (phase, direction, device_index, elapsed_ms, nbytes)
    _handle(subscriber, [(0, 1, 0, 100.0, 10**9), (1, 1, 0, 50.0, 10**9)])
    assert _total_count(_KERNEL_METRIC) == kernel_before + 1
    assert _total_count(_STAGING_METRIC) == staging_before + 1


@pytest.mark.parametrize(
    "sample",
    [
        (0, 1, 0, 100.0),  # wrong arity
        (0, 1, 0, 0.0, 10**9),  # non-positive time
        (0, 1, 0, 100.0, 0),  # non-positive bytes
        (7, 1, 0, 100.0, 10**9),  # unknown phase
    ],
)
def test_malformed_samples_dropped(sample):
    subscriber = TransferPhaseSubscriber()
    kernel_before = _total_count(_KERNEL_METRIC)
    staging_before = _total_count(_STAGING_METRIC)
    _handle(subscriber, [sample])
    assert _total_count(_KERNEL_METRIC) == kernel_before
    assert _total_count(_STAGING_METRIC) == staging_before
