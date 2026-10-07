# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry metrics for the MP coordinator.

Metric names are part of the coordinator's versioned contract: dashboards,
alerts and out-of-tree controllers depend on them, so they are defined here
once. See ``docs/design/v1/mp_coordinator/observability.md``.
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

METER_NAME = "lmcache.mp_coordinator"

KEY_DIRECTORY_PLACEMENTS = "lmcache_coordinator.key_directory.placements"
KEY_DIRECTORY_PLACEMENT_BYTES = "lmcache_coordinator.key_directory.placement_bytes"

# TODO(ruizhang0101): remove the pre-rename names one release after the
# lmcache_coordinator.* names ship.
_LEGACY_KEY_DIRECTORY_PLACEMENTS = "lmcache_mp.key_directory_placement_count"
_LEGACY_KEY_DIRECTORY_PLACEMENT_BYTES = "lmcache_mp.key_directory_placement_size_bytes"


def _per_tier(
    key_directory: "KeyDirectory",
    select: Callable[["DirectoryStats"], tuple[int, int]],
) -> Callable[["CallbackOptions"], list["Observation"]]:
    """Return a gauge callback observing ``select``'s ``(l1, l2)`` pair of
    the directory's stats as one value per tier."""

    def _observe(_options: "CallbackOptions") -> list["Observation"]:
        l1, l2 = select(key_directory.stats())
        return [
            metrics.Observation(l1, {"tier": "l1"}),
            metrics.Observation(l2, {"tier": "l2"}),
        ]

    return _observe


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
        meter = metrics.get_meter(METER_NAME)
    placements = _per_tier(key_directory, lambda s: (s.l1_count, s.l2_count))
    placement_bytes = _per_tier(
        key_directory, lambda s: (s.l1_size_bytes, s.l2_size_bytes)
    )
    meter.create_observable_gauge(
        KEY_DIRECTORY_PLACEMENTS,
        callbacks=[placements],
        description="Placements recorded in the key directory, by tier. A "
        "placement is one place a key is stored: L1 on one server, or one L2 "
        "backend.",
    )
    meter.create_observable_gauge(
        KEY_DIRECTORY_PLACEMENT_BYTES,
        callbacks=[placement_bytes],
        description="Reported logical bytes of the placements recorded in the "
        "key directory, by tier.",
    )
    meter.create_observable_gauge(
        _LEGACY_KEY_DIRECTORY_PLACEMENTS,
        callbacks=[placements],
        description=f"Deprecated: use {KEY_DIRECTORY_PLACEMENTS}.",
    )
    meter.create_observable_gauge(
        _LEGACY_KEY_DIRECTORY_PLACEMENT_BYTES,
        callbacks=[placement_bytes],
        description=f"Deprecated: use {KEY_DIRECTORY_PLACEMENT_BYTES}.",
    )
