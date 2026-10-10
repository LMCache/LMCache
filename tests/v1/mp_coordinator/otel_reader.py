# SPDX-License-Identifier: Apache-2.0
"""Read coordinator metrics in tests through a private OTel provider.

Coordinator components take a ``meter``; tests hand them one from
:func:`private_meter` and read back what they recorded, without touching
the process-global provider (which can be set only once per process).
"""

# Third Party
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def private_meter() -> tuple[Meter, InMemoryMetricReader]:
    """Return a meter on a fresh provider and the reader that collects it."""
    reader = InMemoryMetricReader()
    return MeterProvider(metric_readers=[reader]).get_meter("test"), reader


def read_values(
    reader: InMemoryMetricReader,
) -> dict[str, dict[frozenset[tuple[str, object]], float]]:
    """Collect once: ``{metric: {attributes: value}}``.

    A counter or gauge point reads as its value, a histogram point as its
    count of recordings.
    """
    values: dict[str, dict[frozenset[tuple[str, object]], float]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return values
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                values[metric.name] = {
                    frozenset((point.attributes or {}).items()): (
                        point.count if hasattr(point, "count") else point.value
                    )
                    for point in metric.data.data_points
                }
    return values


def labels(**attributes: object) -> frozenset[tuple[str, object]]:
    """Return the key :func:`read_values` files a point under."""
    return frozenset(attributes.items())
