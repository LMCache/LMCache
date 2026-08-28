# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator metrics initialization."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.v1.mp_coordinator import observability
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory, PlacementStats


def test_disabled_metrics_are_not_initialized() -> None:
    config = MPCoordinatorConfig(metrics_enabled=False)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_not_called()


def test_prometheus_metrics_reuse_coordinator_http_server() -> None:
    config = MPCoordinatorConfig(metrics_enabled=True)

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint=None,
        resource_attributes={"service.name": "lmcache-mp-coordinator"},
        start_http_server=False,
    )


def test_otlp_metrics_reuse_shared_initializer() -> None:
    config = MPCoordinatorConfig(
        metrics_enabled=True,
        otlp_endpoint="http://collector:4317",
    )

    with patch(
        "lmcache.v1.mp_coordinator.observability.init_otel_metrics"
    ) as mock_init:
        init_coordinator_metrics(config)

    mock_init.assert_called_once_with(
        otlp_endpoint="http://collector:4317",
        resource_attributes={"service.name": "lmcache-mp-coordinator"},
        start_http_server=False,
    )


def test_key_directory_gauges_register_once_and_follow_latest_target() -> None:
    first = MagicMock(spec=KeyDirectory)
    first.placement_stats.return_value = PlacementStats(0, 0, 0, 0)
    second = MagicMock(spec=KeyDirectory)
    second.placement_stats.return_value = PlacementStats(2, 300, 1, 400)

    # Isolate the process-global OTel instrument lifecycle from other app tests.
    with (
        patch.object(observability, "_key_directory_metrics_registered", False),
        patch.object(observability, "_key_directory_metrics_target", None),
        patch.object(observability, "register_gauge") as mock_register,
    ):
        observability.register_key_directory_metrics(first)

        assert mock_register.call_count == 2
        count_call, size_call = mock_register.call_args_list
        assert count_call.args[:3] == (
            "lmcache.mp_coordinator",
            "lmcache_mp.key_directory_placement_count",
            "Number of placements currently recorded in the Coordinator "
            "Key Directory, by cache tier.",
        )
        assert size_call.args[:3] == (
            "lmcache.mp_coordinator",
            "lmcache_mp.key_directory_placement_size_bytes",
            "Sum of reported logical object sizes for placements currently "
            "recorded in the Coordinator Key Directory, by cache tier.",
        )

        count_callback = count_call.args[3]
        size_callback = size_call.args[3]
        assert count_callback() == [
            (0, {"tier": "l1"}),
            (0, {"tier": "l2"}),
        ]
        assert size_callback() == [
            (0, {"tier": "l1"}),
            (0, {"tier": "l2"}),
        ]

        observability.register_key_directory_metrics(second)

        assert mock_register.call_count == 2
        assert count_callback() == [
            (2, {"tier": "l1"}),
            (1, {"tier": "l2"}),
        ]
        assert size_callback() == [
            (300, {"tier": "l1"}),
            (400, {"tier": "l2"}),
        ]
