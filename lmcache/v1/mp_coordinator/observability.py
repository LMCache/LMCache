# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics for the MP coordinator.

Metric names are part of the coordinator's versioned contract: dashboards and
alerts depend on them. Every metric is one row of :func:`_metric_specs`, read
at scrape time from a component's ``stats()`` snapshot. See
``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import socket
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.otel_init import init_otel_metrics

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.metrics import CallbackOptions, Meter, Observation

    # First Party
    from lmcache.v1.mp_coordinator.background_loops import (
        BackgroundLoops,
        EventLoopMonitor,
    )
    from lmcache.v1.mp_coordinator.ingest.event_broadcaster import (
        CacheEventBroadcaster,
    )
    from lmcache.v1.mp_coordinator.ingest.event_gate import (
        EventGate,
        InstanceStreamStats,
    )
    from lmcache.v1.mp_coordinator.ingest.kafka_event_source import (
        KafkaCacheEventSource,
    )
    from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
    from lmcache.v1.mp_coordinator.views.key_directory import (
        DirectoryStats,
        KeyDirectory,
    )

# ``(value, attributes)`` per series, as one metric reports at a scrape.
Points = list[tuple[float, dict[str, str]]]


@dataclass(frozen=True)
class CoordinatorMetricSources:
    """The components the coordinator's metrics read.

    Attributes:
        key_directory: Placements and their bytes.
        event_gate: Cache-event loss and freshness, fleet-wide and per server.
        broadcaster: Per-consumer apply time and failures.
        background_loops: Liveness of every periodic loop.
        event_loop_monitor: How long the event loop was blocked.
        registry: Fleet membership and heartbeats.
        controllers_up: Whether each controller, keyed by class name,
            started. The app fills it in as it enters each controller.
        kafka_source: The Kafka cache-event source, or ``None`` when events
            arrive over HTTP (no Kafka metrics then).
    """

    key_directory: "KeyDirectory"
    event_gate: "EventGate"
    broadcaster: "CacheEventBroadcaster"
    background_loops: "BackgroundLoops"
    event_loop_monitor: "EventLoopMonitor"
    registry: "InstanceRegistry"
    controllers_up: Mapping[str, bool]
    kafka_source: "KafkaCacheEventSource | None" = None


def init_coordinator_metrics(config: MPCoordinatorConfig) -> None:
    """Initialize the coordinator's OpenTelemetry metrics pipeline.

    Prometheus pull mode reuses the coordinator's FastAPI server, so this
    function never starts the standalone Prometheus HTTP server.

    Args:
        config: Coordinator configuration controlling metrics export.
    """
    if not config.metrics_enabled:
        return

    init_otel_metrics(
        otlp_endpoint=config.otlp_endpoint,
        resource_attributes={
            "service.name": "lmcache-mp-coordinator",
            # The hostname (the pod name under k8s) tells coordinators
            # apart; the bind address cannot, since every pod binds 0.0.0.0.
            "service.instance.id": socket.gethostname(),
        },
        start_http_server=False,
    )


def register_coordinator_metrics(
    sources: CoordinatorMetricSources, meter: "Meter | None" = None
) -> None:
    """Register every coordinator metric over ``sources``.

    Each instrument is observable: it reads a component's ``stats()`` at
    scrape time, so nothing in the components records into OTel.

    Args:
        sources: The components the metrics read.
        meter: Meter to register on. Defaults to the global provider's
            coordinator meter; tests pass one from a private provider.
    """
    if meter is None:
        meter = metrics.get_meter("lmcache.mp_coordinator")
    for spec in _metric_specs(sources):
        create = (
            meter.create_observable_gauge
            if spec.kind == "gauge"
            else meter.create_observable_counter
        )
        create(
            spec.name,
            callbacks=[_make_observation_callback(spec.read)],
            unit=spec.unit,
            description=spec.description,
        )


@dataclass(frozen=True)
class _MetricSpec:
    """One metric: its name, instrument kind, description and reader."""

    name: str
    kind: Literal["gauge", "counter"]
    description: str
    read: Callable[[], Points]
    unit: str = ""


def _metric_specs(sources: CoordinatorMetricSources) -> list[_MetricSpec]:
    """Return the coordinator's metric catalog, bound to ``sources``."""
    key_directory = sources.key_directory
    event_gate = sources.event_gate
    broadcaster = sources.broadcaster
    loops = sources.background_loops
    prefix = "lmcache_coordinator."
    specs = [
        # -- Key directory ----------------------------------------------------
        _MetricSpec(
            prefix + "key_directory.placements",
            "gauge",
            "Placements recorded in the key directory, by tier. A placement is "
            "one place a key is stored: L1 on one server, or one L2 backend.",
            lambda: _tier_points(key_directory, lambda s: (s.l1_count, s.l2_count)),
        ),
        _MetricSpec(
            prefix + "key_directory.placement_bytes",
            "gauge",
            "Reported logical bytes of the placements recorded in the key "
            "directory, by tier.",
            lambda: _tier_points(
                key_directory, lambda s: (s.l1_size_bytes, s.l2_size_bytes)
            ),
        ),
        # TODO(ruizhang0101): remove the pre-rename names one release after
        # the lmcache_coordinator.* names ship.
        _MetricSpec(
            "lmcache_mp.key_directory_placement_count",
            "gauge",
            "Deprecated: use lmcache_coordinator.key_directory.placements.",
            lambda: _tier_points(key_directory, lambda s: (s.l1_count, s.l2_count)),
        ),
        _MetricSpec(
            "lmcache_mp.key_directory_placement_size_bytes",
            "gauge",
            "Deprecated: use lmcache_coordinator.key_directory.placement_bytes.",
            lambda: _tier_points(
                key_directory, lambda s: (s.l1_size_bytes, s.l2_size_bytes)
            ),
        ),
        # -- Ingest: is the view complete, and how old is it? -----------------
        _MetricSpec(
            prefix + "ingest.event_batches_received",
            "counter",
            "Cache-event batches received from mp servers, by what the gate "
            "did with them.",
            lambda: _batches_received_points(event_gate),
        ),
        _MetricSpec(
            prefix + "ingest.event_batches_missing",
            "counter",
            "Cache-event batches that never arrived: skipped seq values in a "
            "stream the gate was already tracking.",
            lambda: [(event_gate.stats().batches_missing, {})],
        ),
        _MetricSpec(
            prefix + "ingest.events_dropped_by_servers",
            "counter",
            "Cache events the mp servers reported dropping before they reached "
            "the coordinator.",
            lambda: [(event_gate.stats().events_dropped, {})],
        ),
        _MetricSpec(
            prefix + "ingest.server_event_batches_missing",
            "gauge",
            "Cache-event batches from this server that never arrived, in its "
            "current run.",
            lambda: _server_points(event_gate, lambda s: s.missing_batches),
        ),
        _MetricSpec(
            prefix + "ingest.server_events_dropped",
            "gauge",
            "Cache events this server reported dropping before they reached "
            "the coordinator, in its current run.",
            lambda: _server_points(event_gate, lambda s: s.events_dropped),
        ),
        _MetricSpec(
            prefix + "ingest.server_view_incomplete",
            "gauge",
            "1 while the coordinator knows it is missing part of this server's cache.",
            lambda: _server_points(event_gate, lambda s: int(s.gap_detected)),
        ),
        _MetricSpec(
            prefix + "ingest.server_seconds_since_last_event",
            "gauge",
            "Seconds since this server's last cache-event batch was applied.",
            lambda: _server_points(event_gate, lambda s: s.seconds_since_last_event),
        ),
        _MetricSpec(
            prefix + "ingest.server_event_delivery_delay_seconds",
            "gauge",
            "For this server's last applied batch, time from the server "
            "sending it to the coordinator applying it, across the two hosts' "
            "clocks.",
            lambda: _server_points(
                event_gate, lambda s: s.event_delivery_delay_seconds
            ),
            unit="s",
        ),
        _MetricSpec(
            prefix + "ingest.batches_delivered",
            "counter",
            "Admitted cache-event batches handed to each consumer.",
            lambda: [
                (stats.batches_delivered, {"consumer": consumer})
                for consumer, stats in broadcaster.stats().items()
            ],
        ),
        _MetricSpec(
            prefix + "ingest.batch_apply_time_seconds",
            "counter",
            "Time each consumer spent applying admitted batches.",
            lambda: [
                (stats.apply_seconds, {"consumer": consumer})
                for consumer, stats in broadcaster.stats().items()
            ],
        ),
        _MetricSpec(
            prefix + "ingest.batch_apply_failures",
            "counter",
            "Batches (op=consume) or fences (op=fence) a consumer failed to "
            "apply; that consumer now disagrees with the others.",
            lambda: _apply_failure_points(broadcaster),
        ),
        # -- Loops: are they alive? -------------------------------------------
        _MetricSpec(
            prefix + "background_loop.last_success_timestamp_seconds",
            "gauge",
            "Wall-clock time each background loop last finished a pass without "
            "an error.",
            lambda: [
                (stats.last_success_timestamp, {"loop": name})
                for name, stats in loops.stats().items()
                if stats.last_success_timestamp is not None
            ],
        ),
        _MetricSpec(
            prefix + "background_loop.interval_seconds",
            "gauge",
            "Configured seconds between each background loop's passes.",
            lambda: [
                (stats.interval_seconds, {"loop": name})
                for name, stats in loops.stats().items()
            ],
        ),
        _MetricSpec(
            prefix + "background_loop.passes",
            "counter",
            "Background loop passes, by result; a failed pass is logged and "
            "the loop keeps running.",
            lambda: _loop_pass_points(loops),
        ),
        _MetricSpec(
            prefix + "controller.up",
            "gauge",
            "1 if the controller is running, 0 if it failed to start and the "
            "work it does is not happening.",
            lambda: [
                (int(up), {"controller": controller})
                for controller, up in sources.controllers_up.items()
            ],
        ),
        _MetricSpec(
            prefix + "event_loop.blocked_seconds",
            "counter",
            "Total time the coordinator's event loop was blocked.",
            lambda: [(sources.event_loop_monitor.blocked_seconds, {})],
            unit="s",
        ),
        _MetricSpec(
            prefix + "registry.server_seconds_since_heartbeat",
            "gauge",
            "Seconds since each registered server's last heartbeat.",
            lambda: _heartbeat_age_points(sources.registry),
        ),
    ]
    if sources.kafka_source is not None:
        specs.extend(_kafka_specs(sources.kafka_source))
    return specs


def _kafka_specs(source: "KafkaCacheEventSource") -> list[_MetricSpec]:
    """Return the Kafka source's metrics, for a coordinator reading Kafka."""
    prefix = "lmcache_coordinator.kafka."
    return [
        _MetricSpec(
            prefix + "records_read",
            "counter",
            "Cache-event records read from Kafka, by outcome: applied, or "
            "skipped as undecodable or failed.",
            lambda: _records_read_points(source),
        ),
        _MetricSpec(
            prefix + "partitions_assigned",
            "gauge",
            "Partitions of the cache-event topic this coordinator holds; fewer "
            "than the topic has means another consumer-group member holds the "
            "rest.",
            lambda: [(source.stats().partitions_assigned, {})],
        ),
        _MetricSpec(
            prefix + "unread_records",
            "gauge",
            "Cache-event records not yet read, per partition.",
            lambda: [
                (unread, {"partition": str(partition)})
                for partition, unread in source.stats().unread_records.items()
            ],
        ),
    ]


def _make_observation_callback(
    read_points: Callable[[], Points],
) -> Callable[["CallbackOptions"], list["Observation"]]:
    """Return an instrument callback reporting ``read_points``'s values."""

    def _observe(_options: "CallbackOptions") -> list["Observation"]:
        return [
            metrics.Observation(value, attributes)
            for value, attributes in read_points()
        ]

    return _observe


def _tier_points(
    key_directory: "KeyDirectory",
    read_l1_and_l2: Callable[["DirectoryStats"], tuple[int, int]],
) -> Points:
    """Return the directory's ``(l1, l2)`` values as one point per tier."""
    l1_value, l2_value = read_l1_and_l2(key_directory.stats())
    return [(l1_value, {"tier": "l1"}), (l2_value, {"tier": "l2"})]


def _server_points(
    event_gate: "EventGate",
    read_value: Callable[["InstanceStreamStats"], float | None],
) -> Points:
    """Return one point per tracked server, skipping a server whose value is
    ``None`` (not known yet)."""
    points: Points = []
    for instance_id, stream in event_gate.stats().streams.items():
        value = read_value(stream)
        if value is not None:
            points.append((value, {"instance_id": instance_id}))
    return points


def _batches_received_points(event_gate: "EventGate") -> Points:
    """Return the gate's batch counts by what it did with them."""
    stats = event_gate.stats()
    return [
        (stats.batches_applied, {"result": "applied"}),
        (stats.batches_duplicate, {"result": "duplicate"}),
        (stats.batches_stale, {"result": "stale"}),
    ]


def _apply_failure_points(broadcaster: "CacheEventBroadcaster") -> Points:
    """Return each consumer's consume and fence failures."""
    points: Points = []
    for consumer, stats in broadcaster.stats().items():
        points.append((stats.consume_failures, {"consumer": consumer, "op": "consume"}))
        points.append((stats.fence_failures, {"consumer": consumer, "op": "fence"}))
    return points


def _loop_pass_points(loops: "BackgroundLoops") -> Points:
    """Return each loop's passes by result."""
    points: Points = []
    for name, stats in loops.stats().items():
        points.append((stats.passes_ok, {"loop": name, "result": "ok"}))
        points.append((stats.passes_failed, {"loop": name, "result": "failed"}))
    return points


def _heartbeat_age_points(registry: "InstanceRegistry") -> Points:
    """Return the seconds since each registered server's last heartbeat."""
    now = time.monotonic()
    return [
        (now - instance.last_heartbeat_time, {"instance_id": instance.instance_id})
        for instance in registry.all_instances()
    ]


def _records_read_points(source: "KafkaCacheEventSource") -> Points:
    """Return the Kafka source's records by outcome."""
    stats = source.stats()
    return [
        (stats.records_applied, {"result": "applied"}),
        (stats.records_undecodable, {"result": "undecodable"}),
        (stats.records_failed, {"result": "failed"}),
    ]
