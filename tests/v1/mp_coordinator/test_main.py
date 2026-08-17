# SPDX-License-Identifier: Apache-2.0
"""Tests for the legacy MP coordinator module entrypoint."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.v1.mp_coordinator.__main__ import main
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def test_main_initializes_metrics_before_serving() -> None:
    config = MPCoordinatorConfig()
    app = MagicMock()
    calls: list[str] = []

    def record_metrics(_: MPCoordinatorConfig) -> None:
        calls.append("metrics")

    def fake_create_app(_: MPCoordinatorConfig) -> MagicMock:
        calls.append("app")
        return app

    with (
        patch(
            "lmcache.v1.mp_coordinator.__main__.MPCoordinatorConfig.from_env",
            return_value=config,
        ),
        patch(
            "lmcache.v1.mp_coordinator.__main__.init_coordinator_metrics",
            side_effect=record_metrics,
        ) as mock_init_metrics,
        patch(
            "lmcache.v1.mp_coordinator.__main__.create_app",
            side_effect=fake_create_app,
        ),
        patch("lmcache.v1.mp_coordinator.__main__.uvicorn.run") as mock_run,
    ):
        main()

    assert calls == ["metrics", "app"]
    mock_init_metrics.assert_called_once_with(config)
    mock_run.assert_called_once_with(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        timeout_keep_alive=config.timeout_keep_alive,
    )
