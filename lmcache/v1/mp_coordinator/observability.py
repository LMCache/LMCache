# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics for the MP coordinator.

Metric names are part of the coordinator's versioned contract: dashboards and
alerts depend on them. Every metric is defined here, read at scrape time from
a component's ``stats()`` snapshot. A component opts in by calling
``register_metric_source(self)`` (see :mod:`.metric_sources`); this module maps
each registered component to its metrics. See
``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import socket

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_coordinator.background_loops import (
    BackgroundLoops,
    EventLoopMonitor,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.base import ControllerStatus
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
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
from lmcache.v1.mp_coordinator.metric_sources import (
    MetricSource,
    get_and_clear_metric_sources,
)
from lmcache.v1.mp_coordinator.persistence.checkpoint import CheckpointMonitor
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.utils.histogram import Histogram
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.mp_coordinator.views.key_directory import (
    DirectoryStats,
    KeyDirectory,
    LookupStats,
)
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
from lmcache.v1.mp_observability.otel_init import init_otel_metrics

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.metrics import CallbackOptions, Meter, Observation

# ``(value, attributes)`` per series, as one metric reports at a scrape.
Points = list[tuple[float, dict[str, str]]]


_PREFIX = "lmcache_coordinator."


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


def register_coordinator_metrics(meter: "Meter | None" = None) -> None:
    """Register the metrics of every registered component.

    Takes every component registered so far (see :mod:`.metric_sources`),
    so call it after the app has built them all. Each instrument is
    observable: it reads a component's ``stats()`` at scrape time, so
    nothing in the components records into OTel.

    Args:
        meter: Meter to register on. Defaults to the global provider's
            coordinator meter; tests pass one from a private provider.

    Raises:
        TypeError: If a registered component has no metrics defined here.
    """
    if meter is None:
        meter = metrics.get_meter("lmcache.mp_coordinator")
    for source in get_and_clear_metric_sources():
        for spec in _specs_for(source):
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


def _specs_for(source: MetricSource) -> list[_MetricSpec]:
    """Return the metrics of one registered source."""
    match source:
        case KeyDirectory():
            return _key_directory_specs(source)
        case EventGate():
            return _event_gate_specs(source)
        case CacheEventBroadcaster():
            return _broadcaster_specs(source)
        case BackgroundLoops():
            return _background_loop_specs(source)
        case EventLoopMonitor():
            return _event_loop_monitor_specs(source)
        case InstanceRegistry():
            return _instance_registry_specs(source)
        case KafkaCacheEventSource():
            return _kafka_specs(source)
        case ControllerStatus():
            return _controller_status_specs(source)
        case CacheUsageManager():
            return _usage_specs(source)
        case CheckpointMonitor():
            return _checkpoint_specs(source)
        case MetadataPersister():
            return _metadata_specs(source)
        case PrefetchManager():
            return _prefetch_specs(source)
    raise TypeError(f"no metrics are defined for {type(source).__name__}")


def _key_directory_specs(key_directory: KeyDirectory) -> list[_MetricSpec]:
    """Return the key directory's contents, blend index and lookup metrics."""
    return [
        _MetricSpec(
            _PREFIX + "key_directory.keys_tracked",
            "gauge",
            "Keys with at least one placement in the key directory.",
            lambda: [(key_directory.stats().num_keys, {})],
        ),
        _MetricSpec(
            _PREFIX + "key_directory.server_l1_keys",
            "gauge",
            "L1 keys each server reported; a sudden drop to 0 is a fence.",
            lambda: [
                (keys, {"instance_id": instance_id})
                for instance_id, keys in (
                    key_directory.stats().l1_keys_by_instance.items()
                )
            ],
        ),
        _MetricSpec(
            _PREFIX + "key_directory.placements",
            "gauge",
            "Placements recorded in the key directory, by tier. A placement is "
            "one place a key is stored: L1 on one server, or one L2 backend.",
            lambda: _tier_points(key_directory, lambda s: (s.l1_count, s.l2_count)),
        ),
        _MetricSpec(
            _PREFIX + "key_directory.placement_bytes",
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
        _MetricSpec(
            _PREFIX + "blend_index.chunks_indexed",
            "gauge",
            "Chunks in the blend (fragment) index; reported only when blend "
            "lookup is enabled.",
            lambda: _blend_points(key_directory, lambda s: s.blend.num_chunks),
        ),
        _MetricSpec(
            _PREFIX + "blend_index.unique_chunk_contents",
            "gauge",
            "Distinct chunk contents in the blend index; fewer than chunks "
            "means the same content is stored more than once.",
            lambda: _blend_points(key_directory, lambda s: s.blend.num_contents),
        ),
        _MetricSpec(
            _PREFIX + "blend_index.namespaces",
            "gauge",
            "Namespaces (model, cache salt, world size) holding indexed content.",
            lambda: _blend_points(key_directory, lambda s: s.blend.num_namespaces),
        ),
        _MetricSpec(
            _PREFIX + "key_directory.lookups",
            "counter",
            "Lookup requests served, by lookup_type: prefix "
            "(/directory/lookup) or non_prefix (/directory/blend-lookup).",
            lambda: _lookup_points(key_directory, lambda s: s.lookups),
        ),
        _MetricSpec(
            _PREFIX + "key_directory.lookup_chunks_requested",
            "counter",
            "Chunks the lookups asked for.",
            lambda: _lookup_points(key_directory, lambda s: s.chunks_requested),
        ),
        _MetricSpec(
            _PREFIX + "key_directory.lookup_chunks_found",
            "counter",
            "Chunks the lookups found; found / requested is the hit rate.",
            lambda: _lookup_points(key_directory, lambda s: s.chunks_found),
        ),
        *_histogram_specs(
            _PREFIX + "key_directory.lookup_chunks_found_per_request",
            "How many chunks each lookup found.",
            lambda: [
                (stats.chunks_found_per_request, {"lookup_type": lookup_type})
                for lookup_type, stats in key_directory.stats().lookups.items()
            ],
        ),
    ]


def _event_gate_specs(event_gate: EventGate) -> list[_MetricSpec]:
    """Return the gate's metrics: is the view complete, and how old is it?"""
    return [
        _MetricSpec(
            _PREFIX + "ingest.event_batches_received",
            "counter",
            "Cache-event batches received from mp servers, by what the gate "
            "did with them.",
            lambda: _batches_received_points(event_gate),
        ),
        _MetricSpec(
            _PREFIX + "ingest.event_batches_missing",
            "counter",
            "Cache-event batches that never arrived: skipped seq values in a "
            "stream the gate was already tracking.",
            lambda: [(event_gate.stats().batches_missing, {})],
        ),
        _MetricSpec(
            _PREFIX + "ingest.events_dropped_by_servers",
            "counter",
            "Cache events the mp servers reported dropping before they reached "
            "the coordinator.",
            lambda: [(event_gate.stats().events_dropped, {})],
        ),
        _MetricSpec(
            _PREFIX + "ingest.server_event_batches_missing",
            "gauge",
            "Cache-event batches from this server that never arrived, in its "
            "current run.",
            lambda: _server_points(event_gate, lambda s: s.missing_batches),
        ),
        _MetricSpec(
            _PREFIX + "ingest.server_events_dropped",
            "gauge",
            "Cache events this server reported dropping before they reached "
            "the coordinator, in its current run.",
            lambda: _server_points(event_gate, lambda s: s.events_dropped),
        ),
        _MetricSpec(
            _PREFIX + "ingest.server_view_incomplete",
            "gauge",
            "1 while the coordinator knows it is missing part of this server's cache.",
            lambda: _server_points(event_gate, lambda s: int(s.gap_detected)),
        ),
        _MetricSpec(
            _PREFIX + "ingest.server_seconds_since_last_event",
            "gauge",
            "Seconds since this server's last cache-event batch was applied.",
            lambda: _server_points(event_gate, lambda s: s.seconds_since_last_event),
        ),
        _MetricSpec(
            _PREFIX + "ingest.server_event_delivery_delay_seconds",
            "gauge",
            "For this server's last applied batch, time from the server "
            "sending it to the coordinator applying it, across the two hosts' "
            "clocks.",
            lambda: _server_points(
                event_gate, lambda s: s.event_delivery_delay_seconds
            ),
            unit="s",
        ),
    ]


def _broadcaster_specs(broadcaster: CacheEventBroadcaster) -> list[_MetricSpec]:
    """Return each consumer's delivery, apply-time and failure metrics."""
    return [
        _MetricSpec(
            _PREFIX + "ingest.batches_delivered",
            "counter",
            "Admitted cache-event batches handed to each consumer.",
            lambda: [
                (stats.batches_delivered, {"consumer": consumer})
                for consumer, stats in broadcaster.stats().items()
            ],
        ),
        _MetricSpec(
            _PREFIX + "ingest.batch_apply_time_seconds",
            "counter",
            "Time each consumer spent applying admitted batches.",
            lambda: [
                (stats.apply_seconds, {"consumer": consumer})
                for consumer, stats in broadcaster.stats().items()
            ],
        ),
        _MetricSpec(
            _PREFIX + "ingest.batch_apply_failures",
            "counter",
            "Batches (op=consume) or fences (op=fence) a consumer failed to "
            "apply; that consumer now disagrees with the others.",
            lambda: _apply_failure_points(broadcaster),
        ),
    ]


def _background_loop_specs(loops: BackgroundLoops) -> list[_MetricSpec]:
    """Return the metrics telling whether each background loop is alive."""
    return [
        _MetricSpec(
            _PREFIX + "background_loop.last_success_timestamp_seconds",
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
            _PREFIX + "background_loop.interval_seconds",
            "gauge",
            "Configured seconds between each background loop's passes.",
            lambda: [
                (stats.interval_seconds, {"loop": name})
                for name, stats in loops.stats().items()
            ],
        ),
        _MetricSpec(
            _PREFIX + "background_loop.passes",
            "counter",
            "Background loop passes, by result; a failed pass is logged and "
            "the loop keeps running.",
            lambda: _loop_pass_points(loops),
        ),
    ]


def _event_loop_monitor_specs(monitor: EventLoopMonitor) -> list[_MetricSpec]:
    """Return how long the event loop was blocked."""
    return [
        _MetricSpec(
            _PREFIX + "event_loop.blocked_seconds",
            "counter",
            "Total time the coordinator's event loop was blocked.",
            lambda: [(monitor.stats().blocked_seconds, {})],
            unit="s",
        ),
    ]


def _instance_registry_specs(registry: InstanceRegistry) -> list[_MetricSpec]:
    """Return fleet membership, its churn, and each server's heartbeat age."""
    return [
        _MetricSpec(
            _PREFIX + "registry.servers_registered",
            "gauge",
            "Number of mp servers registered with the coordinator.",
            lambda: [(registry.stats().servers_registered, {})],
        ),
        _MetricSpec(
            _PREFIX + "registry.server_membership_changes",
            "counter",
            "Fleet membership changes, by change: registered, re_registered, "
            "deregistered, timed_out, unknown_heartbeat. A climbing "
            "re_registered or timed_out means servers are flapping.",
            lambda: [
                (count, {"change": change})
                for change, count in registry.stats().membership_changes.items()
            ],
        ),
        _MetricSpec(
            _PREFIX + "registry.server_seconds_since_heartbeat",
            "gauge",
            "Seconds since each registered server's last heartbeat.",
            lambda: [
                (seconds, {"instance_id": instance_id})
                for instance_id, seconds in (
                    registry.stats().seconds_since_heartbeat.items()
                )
            ],
        ),
    ]


def _controller_status_specs(status: ControllerStatus) -> list[_MetricSpec]:
    """Return whether each controller started."""
    return [
        _MetricSpec(
            _PREFIX + "controller.up",
            "gauge",
            "1 if the controller is running, 0 if it failed to start and the "
            "work it does is not happening.",
            lambda: [
                (int(up), {"controller": controller})
                for controller, up in status.stats().items()
            ],
        ),
    ]


def _usage_specs(usage: CacheUsageManager) -> list[_MetricSpec]:
    """Return the fleet's bytes per tier."""
    return [
        _MetricSpec(
            _PREFIX + "usage.bytes_stored",
            "gauge",
            "Reported bytes the fleet holds, by tier.",
            lambda: [
                (n_bytes, {"tier": tier})
                for tier, n_bytes in usage.stats().bytes_stored.items()
            ],
        ),
    ]


def _checkpoint_specs(monitor: CheckpointMonitor) -> list[_MetricSpec]:
    """Return the checkpoint file's metrics."""
    return [
        _MetricSpec(
            _PREFIX + "checkpoint_file.last_write_timestamp_seconds",
            "gauge",
            "Wall-clock time the last good checkpoint was written.",
            lambda: _present_points(monitor.stats().last_write_timestamp),
        ),
        _MetricSpec(
            _PREFIX + "checkpoint_file.writes",
            "counter",
            "Checkpoint writes, by result: ok, quiesce_timeout (a batch was "
            "still being applied), error.",
            lambda: [
                (count, {"result": result})
                for result, count in monitor.stats().writes.items()
            ],
        ),
        *_histogram_specs(
            _PREFIX + "checkpoint_file.write_duration_seconds",
            "Time per checkpoint write phase: pause_ingest, copy (how long "
            "ingest was paused), write.",
            lambda: [
                (histogram, {"phase": phase})
                for phase, histogram in (monitor.stats().write_duration_seconds.items())
            ],
        ),
        _MetricSpec(
            _PREFIX + "checkpoint_file.size_bytes",
            "gauge",
            "Size of the last good checkpoint.",
            lambda: _present_points(monitor.stats().size_bytes),
        ),
        _MetricSpec(
            _PREFIX + "checkpoint_file.restored_on_start",
            "gauge",
            "1 for what startup did with the checkpoint: cold (none), "
            "restored, or ignored (unreadable).",
            lambda: _restore_points(monitor),
        ),
    ]


def _metadata_specs(persister: MetadataPersister) -> list[_MetricSpec]:
    """Return the metadata file's metrics."""
    return [
        _MetricSpec(
            _PREFIX + "metadata_file.writes",
            "counter",
            "Metadata (pin and quota) saves, by result; an error is a change "
            "that will not survive a restart.",
            lambda: [
                (count, {"result": result})
                for result, count in persister.stats().writes.items()
            ],
        ),
    ]


def _prefetch_specs(prefetch: PrefetchManager) -> list[_MetricSpec]:
    """Return the warm-prefetch relay's metrics."""
    return [
        _MetricSpec(
            _PREFIX + "prefetch.requests_submitted",
            "counter",
            "Prefetches asked for, by result: submitted, noop (shorter than a "
            "chunk), unknown_server, server_error.",
            lambda: [
                (count, {"result": result})
                for result, count in prefetch.stats().requests_submitted.items()
            ],
        ),
        *_histogram_specs(
            _PREFIX + "prefetch.chunks_requested_per_request",
            "Chunks each prefetch asked to warm.",
            lambda: [(prefetch.stats().chunks_requested_per_request, {})],
        ),
        *_histogram_specs(
            _PREFIX + "prefetch.submit_duration_seconds",
            "Time for the mp server to accept a prefetch.",
            lambda: [(prefetch.stats().submit_duration_seconds, {})],
        ),
        _MetricSpec(
            _PREFIX + "prefetch.requests_in_flight",
            "gauge",
            "Prefetches submitted to each server and not yet seen completed.",
            lambda: [
                (count, {"instance_id": instance_id})
                for instance_id, count in prefetch.stats().requests_in_flight.items()
            ],
        ),
        *_histogram_specs(
            _PREFIX + "prefetch.completion_duration_seconds",
            "Submit to the first poll that saw the prefetch completed; "
            "includes the caller's poll interval.",
            lambda: [(prefetch.stats().completion_duration_seconds, {})],
        ),
        _MetricSpec(
            _PREFIX + "prefetch.keys_requested",
            "counter",
            "Keys the completed prefetches asked to warm.",
            lambda: [(prefetch.stats().keys_requested, {})],
        ),
        _MetricSpec(
            _PREFIX + "prefetch.keys_found",
            "counter",
            "Keys found in L2 and loaded into L1; found / requested is the "
            "prefetch hit rate.",
            lambda: [(prefetch.stats().keys_found, {})],
        ),
        _MetricSpec(
            _PREFIX + "prefetch.status_polls",
            "counter",
            "Prefetch status polls, by result: pending, completed, "
            "unknown_request, server_error.",
            lambda: [
                (count, {"result": result})
                for result, count in prefetch.stats().status_polls.items()
            ],
        ),
        _MetricSpec(
            _PREFIX + "prefetch.requests_abandoned",
            "counter",
            "Prefetches never polled to completion within 10 minutes: the "
            "caller gave up, or the server left.",
            lambda: [(prefetch.stats().requests_abandoned, {})],
        ),
    ]


def _kafka_specs(source: KafkaCacheEventSource) -> list[_MetricSpec]:
    """Return the Kafka source's metrics, for a coordinator reading Kafka."""
    return [
        _MetricSpec(
            _PREFIX + "kafka.records_read",
            "counter",
            "Cache-event records read from Kafka, by outcome: applied, or "
            "skipped as undecodable or failed.",
            lambda: _records_read_points(source),
        ),
        _MetricSpec(
            _PREFIX + "kafka.partitions_assigned",
            "gauge",
            "Partitions of the cache-event topic this coordinator holds; fewer "
            "than the topic has means another consumer-group member holds the "
            "rest.",
            lambda: [(source.stats().partitions_assigned, {})],
        ),
        _MetricSpec(
            _PREFIX + "kafka.unread_records",
            "gauge",
            "Cache-event records not yet read, per partition.",
            lambda: [
                (unread, {"partition": str(partition)})
                for partition, unread in source.stats().unread_records.items()
            ],
        ),
    ]


def _histogram_specs(
    name: str,
    description: str,
    read_histograms: Callable[[], list[tuple[Histogram, dict[str, str]]]],
) -> list[_MetricSpec]:
    """Return the ``_bucket``, ``_sum`` and ``_count`` counters of a
    histogram a component keeps in its ``stats()``.

    ``_bucket`` is cumulative and labelled ``le``, as Prometheus expects,
    so ``histogram_quantile`` works on it.
    """
    return [
        _MetricSpec(
            name + "_bucket",
            "counter",
            description + " Cumulative bucket counts, labelled le.",
            lambda: [
                (count, {**attributes, "le": le})
                for histogram, attributes in read_histograms()
                for le, count in histogram.cumulative_buckets()
            ],
        ),
        _MetricSpec(
            name + "_sum",
            "counter",
            description + " Sum of the observed values.",
            lambda: [
                (histogram.sum, attributes)
                for histogram, attributes in read_histograms()
            ],
        ),
        _MetricSpec(
            name + "_count",
            "counter",
            description + " Number of observations.",
            lambda: [
                (histogram.count, attributes)
                for histogram, attributes in read_histograms()
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
    key_directory: KeyDirectory,
    read_l1_and_l2: Callable[[DirectoryStats], tuple[int, int]],
) -> Points:
    """Return the directory's ``(l1, l2)`` values as one point per tier."""
    l1_value, l2_value = read_l1_and_l2(key_directory.stats())
    return [(l1_value, {"tier": "l1"}), (l2_value, {"tier": "l2"})]


def _blend_points(
    key_directory: KeyDirectory, read_value: Callable[[DirectoryStats], int]
) -> Points:
    """Return the blend index value, or nothing while blend lookup is off."""
    directory_stats = key_directory.stats()
    if not directory_stats.blend_lookup_enabled:
        return []
    return [(read_value(directory_stats), {})]


def _lookup_points(
    key_directory: KeyDirectory, read_value: Callable[[LookupStats], int]
) -> Points:
    """Return one point per lookup type."""
    return [
        (read_value(stats), {"lookup_type": lookup_type})
        for lookup_type, stats in key_directory.stats().lookups.items()
    ]


def _present_points(value: float | None) -> Points:
    """Return ``value`` as one point, or nothing while it is unknown."""
    return [] if value is None else [(value, {})]


def _restore_points(monitor: CheckpointMonitor) -> Points:
    """Return 1 for what startup did with the checkpoint, 0 for the rest."""
    outcome = monitor.stats().restored_on_start
    if outcome is None:
        return []
    return [
        (int(candidate == outcome), {"outcome": candidate})
        for candidate in ("cold", "restored", "ignored")
    ]


def _server_points(
    event_gate: EventGate,
    read_value: Callable[[InstanceStreamStats], float | None],
) -> Points:
    """Return one point per tracked server, skipping a server whose value is
    ``None`` (not known yet)."""
    points: Points = []
    for instance_id, stream in event_gate.stats().streams.items():
        value = read_value(stream)
        if value is not None:
            points.append((value, {"instance_id": instance_id}))
    return points


def _batches_received_points(event_gate: EventGate) -> Points:
    """Return the gate's batch counts by what it did with them."""
    stats = event_gate.stats()
    return [
        (stats.batches_applied, {"result": "applied"}),
        (stats.batches_duplicate, {"result": "duplicate"}),
        (stats.batches_stale, {"result": "stale"}),
    ]


def _apply_failure_points(broadcaster: CacheEventBroadcaster) -> Points:
    """Return each consumer's consume and fence failures."""
    points: Points = []
    for consumer, stats in broadcaster.stats().items():
        points.append((stats.consume_failures, {"consumer": consumer, "op": "consume"}))
        points.append((stats.fence_failures, {"consumer": consumer, "op": "fence"}))
    return points


def _loop_pass_points(loops: BackgroundLoops) -> Points:
    """Return each loop's passes by result."""
    points: Points = []
    for name, stats in loops.stats().items():
        points.append((stats.passes_ok, {"loop": name, "result": "ok"}))
        points.append((stats.passes_failed, {"loop": name, "result": "failed"}))
    return points


def _records_read_points(source: KafkaCacheEventSource) -> Points:
    """Return the Kafka source's records by outcome."""
    stats = source.stats()
    return [
        (stats.records_applied, {"result": "applied"}),
        (stats.records_undecodable, {"result": "undecodable"}),
        (stats.records_failed, {"result": "failed"}),
    ]
