# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics initialization for the MP coordinator."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.otel_init import init_otel_metrics, register_gauge

if TYPE_CHECKING:
    # First Party
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
