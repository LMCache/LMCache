# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics initialization for the MP coordinator."""

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.otel_init import init_otel_metrics, register_gauge

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
    from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

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
        resource_attributes={"service.name": "lmcache-mp-coordinator"},
        start_http_server=False,
    )


def register_key_directory_metrics(key_directory: "KeyDirectory") -> None:
    """Register gauges for the current Key Directory placement state.

    The callbacks remain bound to the directory owned by the coordinator app.

    Args:
        key_directory: The actual discovered directory used by the coordinator.

    Returns:
        None.
    """
    register_gauge(
        _METER_NAME,
        "lmcache_mp.key_directory_placement_count",
        "Number of placements currently recorded in the Coordinator Key Directory, "
        "by cache tier.",
        lambda: _placement_count_observations(key_directory),
    )
    register_gauge(
        _METER_NAME,
        "lmcache_mp.key_directory_placement_size_bytes",
        "Sum of reported logical object sizes for placements currently recorded in "
        "the Coordinator Key Directory, by cache tier.",
        lambda: _placement_size_observations(key_directory),
    )


def _placement_count_observations(
    key_directory: "KeyDirectory",
) -> list[tuple[int | float, dict[str, object]]]:
    """Return fixed-cardinality placement-count observations."""
    stats = key_directory.stats()
    return [
        (stats.l1_count, {"tier": "l1"}),
        (stats.l2_count, {"tier": "l2"}),
    ]


def _placement_size_observations(
    key_directory: "KeyDirectory",
) -> list[tuple[int | float, dict[str, object]]]:
    """Return fixed-cardinality placement-size observations."""
    stats = key_directory.stats()
    return [
        (stats.l1_size_bytes, {"tier": "l1"}),
        (stats.l2_size_bytes, {"tier": "l2"}),
    ]


def register_event_gate_metrics(event_gate: "EventGate") -> None:
    """Register per-instance gauges for cache-event loss at the gate.

    Each covers the emitter's current incarnation, since this coordinator
    started tracking it (see ``InstanceStreamStats``).

    Args:
        event_gate: The gate admitting the coordinator's cache events.
    """

    def _per_instance(
        field: str,
    ) -> Callable[[], list[tuple[int | float, dict[str, object]]]]:
        def _observe() -> list[tuple[int | float, dict[str, object]]]:
            return [
                (getattr(stream, field), {"instance_id": instance_id})
                for instance_id, stream in event_gate.stats().items()
            ]

        return _observe

    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_event_loss_incidents_total",
        "Admitted cache-event batches showing loss (a seq gap or a rise in "
        "the emitter-reported lost count), per emitter. Counts incidents, "
        "not events.",
        _per_instance("loss_incidents_total"),
    )
    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_event_lost_events_total",
        "Cache events lost before reaching the coordinator, per emitter. "
        "Counts only loss the emitter reported.",
        _per_instance("lost_events_total"),
    )
    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_event_admitted_events_total",
        "Cache events (batch entries) admitted, per emitter.",
        _per_instance("admitted_events_total"),
    )
