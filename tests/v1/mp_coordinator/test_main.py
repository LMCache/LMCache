# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP coordinator module entrypoint."""

# Standard
from unittest.mock import MagicMock, patch
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.__main__ import main
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _serve(argv: list[str]) -> tuple[list[str], MPCoordinatorConfig, MagicMock]:
    """Run ``main()`` with ``argv``, returning the call order, config and app."""
    app = MagicMock()
    calls: list[str] = []
    captured: dict[str, MPCoordinatorConfig] = {}

    def record_metrics(config: MPCoordinatorConfig) -> None:
        calls.append("metrics")
        captured["config"] = config

    def fake_create_app(config: MPCoordinatorConfig) -> MagicMock:
        calls.append("app")
        return app

    with (
        patch.object(sys, "argv", ["lmcache.v1.mp_coordinator", *argv]),
        patch(
            "lmcache.v1.mp_coordinator.observability.init_coordinator_metrics",
            side_effect=record_metrics,
        ),
        patch(
            "lmcache.v1.mp_coordinator.app.create_app",
            side_effect=fake_create_app,
        ),
        patch("uvicorn.run") as mock_run,
    ):
        main()
    return calls, captured["config"], mock_run


def test_main_initializes_metrics_before_serving() -> None:
    calls, config, mock_run = _serve([])

    assert calls == ["metrics", "app"]
    assert config == MPCoordinatorConfig()
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs == {
        "host": config.host,
        "port": config.port,
        "log_level": "info",
        "timeout_keep_alive": config.timeout_keep_alive,
    }


def test_main_applies_flags() -> None:
    """The module entrypoint accepts the same flags as ``lmcache coordinator``."""
    _, config, _ = _serve(["--port", "9999", "--chunk-size", "512"])

    assert config.port == 9999
    assert config.chunk_size == 512
    # Unset flags keep the config defaults.
    assert config.host == MPCoordinatorConfig.host


def test_main_rejects_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        _serve(["--not-a-flag"])
