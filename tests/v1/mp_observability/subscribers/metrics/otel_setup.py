# SPDX-License-Identifier: Apache-2.0

"""Shared OTel MeterProvider for all metrics subscriber tests.

OTel only allows one MeterProvider per process.  This module sets it up
once so that every test file in this directory reads from the same reader.

Import as::

    from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader
"""

# Third Party
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

reader = InMemoryMetricReader()
_provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(_provider)
