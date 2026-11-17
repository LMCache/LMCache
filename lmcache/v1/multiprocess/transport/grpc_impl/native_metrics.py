# SPDX-License-Identifier: Apache-2.0
"""gRPC server metrics for the native server mode.

grpcio-observability can't see calls that the native front serves, so this
turns the front's own totals into gRPC's ``grpc.server.call.started`` and
``grpc.server.call.duration``, named and labeled the way the plugin exports
them for the sync server.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import time

# Third Party
import grpc

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.sdk.metrics.export import Metric

# Same descriptions as grpcio-observability, so both modes share one family.
_STARTED_DESCRIPTION = "Number of server calls started"
_DURATION_DESCRIPTION = (
    "End-to-end time taken to complete a call from server transport's perspective"
)
_STATUS_NAMES = {code.value[0]: code.name for code in grpc.StatusCode}


def native_metrics_source(front: Any) -> Callable[[], list["Metric"]]:
    """Build a metrics source over a ``NativeGrpcFront`` created with metrics.

    Args:
        front: A pygrpc-native ``NativeGrpcFront`` with ``metrics=True``.

    Returns:
        A callable for ``register_metrics_source`` that reads the front's
        totals on each collection.
    """
    # Third Party
    from opentelemetry.sdk.metrics.export import (
        AggregationTemporality,
        Histogram,
        HistogramDataPoint,
        Metric,
        NumberDataPoint,
        Sum,
    )

    def source() -> list[Metric]:
        snapshot = front.metrics()
        if snapshot is None:
            return []
        start = snapshot["start_time_unix_nano"]
        now = time.time_ns()
        started = []
        durations = []
        for path, totals in snapshot["methods"].items():
            if not totals["started"]:
                continue
            # grpcio reports the method without the leading slash.
            method = path.lstrip("/")
            started.append(
                NumberDataPoint(
                    attributes={"grpc.method": method},
                    start_time_unix_nano=start,
                    time_unix_nano=now,
                    value=totals["started"],
                )
            )
            for code, histogram in totals["status"].items():
                durations.append(
                    HistogramDataPoint(
                        attributes={
                            "grpc.method": method,
                            "grpc.status": _STATUS_NAMES.get(code, str(code)),
                        },
                        start_time_unix_nano=start,
                        time_unix_nano=now,
                        count=histogram["count"],
                        sum=histogram["sum"],
                        bucket_counts=histogram["buckets"],
                        explicit_bounds=snapshot["bounds"],
                        min=histogram["min"],
                        max=histogram["max"],
                    )
                )
        if not started:
            return []
        return [
            Metric(
                name="grpc.server.call.started",
                description=_STARTED_DESCRIPTION,
                unit="{call}",
                data=Sum(
                    data_points=started,
                    aggregation_temporality=AggregationTemporality.CUMULATIVE,
                    is_monotonic=True,
                ),
            ),
            Metric(
                name="grpc.server.call.duration",
                description=_DURATION_DESCRIPTION,
                unit="s",
                data=Histogram(
                    data_points=durations,
                    aggregation_temporality=AggregationTemporality.CUMULATIVE,
                ),
            ),
        ]

    return source
