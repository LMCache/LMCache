# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics for the MP coordinator.

Metric names are part of the coordinator's versioned contract: dashboards and
alerts depend on them. See ``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING
import socket

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.otel_init import init_otel_metrics

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.metrics import CallbackOptions, Meter, Observation

    # First Party
    from lmcache.v1.mp_coordinator.ingest.event_broadcaster import (
        CacheEventBroadcaster,
    )
    from lmcache.v1.mp_coordinator.ingest.event_gate import (
        EventGate,
        InstanceStreamStats,
    )
    from lmcache.v1.mp_coordinator.views.key_directory import (
        DirectoryStats,
        KeyDirectory,
    )


# The meter every coordinator instrument is registered on.
_METER_NAME = "lmcache.mp_coordinator"


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


def register_key_directory_metrics(
    key_directory: "KeyDirectory", meter: "Meter | None" = None
) -> None:
    """Register gauges for the current Key Directory placement state.

    The callbacks remain bound to the directory owned by the coordinator app.
    The pre-rename ``lmcache_mp.*`` names are emitted alongside for one
    release, so existing dashboards keep working while they migrate.

    Args:
        key_directory: The actual discovered directory used by the coordinator.
        meter: Meter to register on. Defaults to the global provider's
            coordinator meter; tests pass one from a private provider.
    """
    if meter is None:
        meter = metrics.get_meter(_METER_NAME)
    placements_callback = _make_tier_gauge_callback(
        key_directory, lambda stats: (stats.l1_count, stats.l2_count)
    )
    placement_bytes_callback = _make_tier_gauge_callback(
        key_directory, lambda stats: (stats.l1_size_bytes, stats.l2_size_bytes)
    )
    meter.create_observable_gauge(
        "lmcache_coordinator.key_directory.placements",
        callbacks=[placements_callback],
        description="Placements recorded in the key directory, by tier. A "
        "placement is one place a key is stored: L1 on one server, or one L2 "
        "backend.",
    )
    meter.create_observable_gauge(
        "lmcache_coordinator.key_directory.placement_bytes",
        callbacks=[placement_bytes_callback],
        description="Reported logical bytes of the placements recorded in the "
        "key directory, by tier.",
    )
    # TODO(ruizhang0101): remove the pre-rename names one release after the
    # lmcache_coordinator.* names ship.
    meter.create_observable_gauge(
        "lmcache_mp.key_directory_placement_count",
        callbacks=[placements_callback],
        description="Deprecated: use lmcache_coordinator.key_directory.placements.",
    )
    meter.create_observable_gauge(
        "lmcache_mp.key_directory_placement_size_bytes",
        callbacks=[placement_bytes_callback],
        description="Deprecated: use "
        "lmcache_coordinator.key_directory.placement_bytes.",
    )


def register_event_gate_metrics(
    event_gate: "EventGate", meter: "Meter | None" = None
) -> None:
    """Register the gate's loss metrics: fleet counters and per-server gauges.

    Both read :meth:`EventGate.stats`. The counters cover this process's
    lifetime. The gauges give one series per
    emitter the gate tracks, gone when its cursor goes (departure or
    timeout), which is why per-server values are gauges rather than
    counters. Each covers the emitter's current incarnation.

    Args:
        event_gate: The gate admitting the coordinator's cache events.
        meter: Meter to register on. Defaults to the global provider's
            coordinator meter; tests pass one from a private provider.
    """
    if meter is None:
        meter = metrics.get_meter(_METER_NAME)

    def read_batches_received() -> list[tuple[float, dict[str, str]]]:
        stats = event_gate.stats()
        return [
            (stats.batches_applied, {"result": "applied"}),
            (stats.batches_duplicate, {"result": "duplicate"}),
            (stats.batches_stale, {"result": "stale"}),
        ]

    meter.create_observable_counter(
        "lmcache_coordinator.ingest.event_batches_received",
        callbacks=[_make_observation_callback(read_batches_received)],
        description="Cache-event batches received from mp servers, by what "
        "the gate did with them.",
    )
    meter.create_observable_counter(
        "lmcache_coordinator.ingest.event_batches_missing",
        callbacks=[
            _make_observation_callback(
                lambda: [(event_gate.stats().batches_missing, {})]
            )
        ],
        description="Cache-event batches that never arrived: skipped seq "
        "values in a stream the gate was already tracking.",
    )
    meter.create_observable_counter(
        "lmcache_coordinator.ingest.events_dropped_by_servers",
        callbacks=[
            _make_observation_callback(
                lambda: [(event_gate.stats().events_dropped, {})]
            )
        ],
        description="Cache events the mp servers reported dropping before "
        "they reached the coordinator.",
    )
    meter.create_observable_gauge(
        "lmcache_coordinator.ingest.server_event_batches_missing",
        callbacks=[
            _make_server_gauge_callback(
                event_gate, lambda stream: stream.missing_batches
            )
        ],
        description="Cache-event batches from this server that never arrived, "
        "in its current run.",
    )
    meter.create_observable_gauge(
        "lmcache_coordinator.ingest.server_events_dropped",
        callbacks=[
            _make_server_gauge_callback(
                event_gate, lambda stream: stream.events_dropped
            )
        ],
        description="Cache events this server reported dropping before they "
        "reached the coordinator, in its current run.",
    )
    meter.create_observable_gauge(
        "lmcache_coordinator.ingest.server_view_incomplete",
        callbacks=[
            _make_server_gauge_callback(
                event_gate, lambda stream: int(stream.gap_detected)
            )
        ],
        description="1 while the coordinator knows it is missing part of this "
        "server's cache.",
    )


def register_broadcaster_metrics(
    broadcaster: "CacheEventBroadcaster", meter: "Meter | None" = None
) -> None:
    """Register per-consumer counters over the broadcaster's tallies.

    Every view and controller that consumes cache events gets the same
    three series, labelled by its class name: batches delivered, time spent
    applying them (apply time / batches delivered is the average per
    batch), and failures. A failure leaves that consumer disagreeing with
    the others.

    Args:
        broadcaster: The broadcaster fanning batches out to the consumers.
        meter: Meter to register on. Defaults to the global provider's
            coordinator meter; tests pass one from a private provider.
    """
    if meter is None:
        meter = metrics.get_meter(_METER_NAME)

    def read_batches_delivered() -> list[tuple[float, dict[str, str]]]:
        return [
            (stats.batches_delivered, {"consumer": consumer})
            for consumer, stats in broadcaster.stats().items()
        ]

    def read_apply_seconds() -> list[tuple[float, dict[str, str]]]:
        return [
            (stats.apply_seconds, {"consumer": consumer})
            for consumer, stats in broadcaster.stats().items()
        ]

    def read_apply_failures() -> list[tuple[float, dict[str, str]]]:
        points: list[tuple[float, dict[str, str]]] = []
        for consumer, stats in broadcaster.stats().items():
            points.append(
                (stats.consume_failures, {"consumer": consumer, "op": "consume"})
            )
            points.append((stats.fence_failures, {"consumer": consumer, "op": "fence"}))
        return points

    meter.create_observable_counter(
        "lmcache_coordinator.ingest.batches_delivered",
        callbacks=[_make_observation_callback(read_batches_delivered)],
        description="Admitted cache-event batches handed to each consumer.",
    )
    meter.create_observable_counter(
        "lmcache_coordinator.ingest.batch_apply_time_seconds",
        callbacks=[_make_observation_callback(read_apply_seconds)],
        description="Time each consumer spent applying admitted batches.",
    )
    meter.create_observable_counter(
        "lmcache_coordinator.ingest.batch_apply_failures",
        callbacks=[_make_observation_callback(read_apply_failures)],
        description="Batches (op=consume) or fences (op=fence) a consumer "
        "failed to apply; that consumer now disagrees with the others.",
    )


def _make_observation_callback(
    read_points: Callable[[], list[tuple[float, dict[str, str]]]],
) -> Callable[["CallbackOptions"], list["Observation"]]:
    """Return an instrument callback reporting ``read_points``'s values.

    Args:
        read_points: Returns ``(value, attributes)`` pairs, one per series.

    Returns:
        A callback observing each pair as one series.
    """

    def _observe(_options: "CallbackOptions") -> list["Observation"]:
        return [
            metrics.Observation(value, attributes)
            for value, attributes in read_points()
        ]

    return _observe


def _make_tier_gauge_callback(
    key_directory: "KeyDirectory",
    read_l1_and_l2: Callable[["DirectoryStats"], tuple[int, int]],
) -> Callable[["CallbackOptions"], list["Observation"]]:
    """Return a gauge callback that reports one value per cache tier.

    Args:
        key_directory: The directory whose stats the callback reads.
        read_l1_and_l2: Picks the ``(l1, l2)`` values out of the stats.

    Returns:
        A callback observing the L1 value under ``tier="l1"`` and the L2
        value under ``tier="l2"``.
    """

    def _observe_tiers(_options: "CallbackOptions") -> list["Observation"]:
        l1_value, l2_value = read_l1_and_l2(key_directory.stats())
        return [
            metrics.Observation(l1_value, {"tier": "l1"}),
            metrics.Observation(l2_value, {"tier": "l2"}),
        ]

    return _observe_tiers


def _make_server_gauge_callback(
    event_gate: "EventGate",
    read_value: Callable[["InstanceStreamStats"], int],
) -> Callable[["CallbackOptions"], list["Observation"]]:
    """Return a gauge callback that reports one value per mp server.

    Args:
        event_gate: The gate whose stream cursors the callback reads.
        read_value: Picks the value to report out of one server's cursor.

    Returns:
        A callback observing each tracked server's value under its
        ``instance_id``; a server the gate stopped tracking is not reported.
    """

    def _observe_servers(_options: "CallbackOptions") -> list["Observation"]:
        return [
            metrics.Observation(read_value(stream), {"instance_id": instance_id})
            for instance_id, stream in event_gate.stats().streams.items()
        ]

    return _observe_servers
