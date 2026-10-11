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
    from lmcache.v1.mp_coordinator.views.key_directory import (
        DirectoryStats,
        KeyDirectory,
    )


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
        meter = metrics.get_meter("lmcache.mp_coordinator")
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
