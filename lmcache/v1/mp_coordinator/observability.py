# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics initialization for the MP coordinator."""

# Standard
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.otel_init import init_otel_metrics, register_gauge

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

_METER_NAME = "lmcache.mp_coordinator"
_PLACEMENT_COUNT_NAME = "lmcache_mp.key_directory_placement_count"
_PLACEMENT_COUNT_DESCRIPTION = (
    "Number of placements currently recorded in the Coordinator Key Directory, "
    "by cache tier."
)
_PLACEMENT_SIZE_NAME = "lmcache_mp.key_directory_placement_size_bytes"
_PLACEMENT_SIZE_DESCRIPTION = (
    "Sum of reported logical object sizes for placements currently recorded in "
    "the Coordinator Key Directory, by cache tier."
)

# The OTel MeterProvider is process-global and does not support unregistering an
# instrument. Keep one pair of gauges and redirect their callbacks to the most
# recently constructed coordinator app, matching the existing L1Manager pattern.
_key_directory_metrics_lock = threading.Lock()
_key_directory_metrics_registered = False
_key_directory_metrics_target: "KeyDirectory | None" = None


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

    The instruments are registered once per process. Later calls update their
    target so test-created or replacement coordinator apps do not leave the
    callbacks bound to an obsolete directory.

    Args:
        key_directory: The actual discovered directory used by the coordinator.

    Returns:
        None.
    """
    global _key_directory_metrics_registered, _key_directory_metrics_target

    with _key_directory_metrics_lock:
        _key_directory_metrics_target = key_directory
        if _key_directory_metrics_registered:
            return
        _key_directory_metrics_registered = True

    register_gauge(
        _METER_NAME,
        _PLACEMENT_COUNT_NAME,
        _PLACEMENT_COUNT_DESCRIPTION,
        _placement_count_observations,
    )
    register_gauge(
        _METER_NAME,
        _PLACEMENT_SIZE_NAME,
        _PLACEMENT_SIZE_DESCRIPTION,
        _placement_size_observations,
    )


def _placement_count_observations() -> list[tuple[int | float, dict[str, object]]]:
    """Return fixed-cardinality placement-count observations."""
    target = _key_directory_metrics_current_target()
    if target is None:
        return [(0, {"tier": "l1"}), (0, {"tier": "l2"})]
    stats = target.placement_stats()
    return [
        (stats.l1_count, {"tier": "l1"}),
        (stats.l2_count, {"tier": "l2"}),
    ]


def _placement_size_observations() -> list[tuple[int | float, dict[str, object]]]:
    """Return fixed-cardinality placement-size observations."""
    target = _key_directory_metrics_current_target()
    if target is None:
        return [(0, {"tier": "l1"}), (0, {"tier": "l2"})]
    stats = target.placement_stats()
    return [
        (stats.l1_size_bytes, {"tier": "l1"}),
        (stats.l2_size_bytes, {"tier": "l2"}),
    ]


def _key_directory_metrics_current_target() -> "KeyDirectory | None":
    """Copy the current target without taking the directory's own lock."""
    with _key_directory_metrics_lock:
        return _key_directory_metrics_target
