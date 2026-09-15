# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def test_defaults_are_valid() -> None:
    config = MPCoordinatorConfig()
    assert config.instance_timeout > 0
    assert config.metrics_enabled is True
    assert config.otlp_endpoint is None


def test_non_positive_intervals_rejected():
    with pytest.raises(ValueError):
        MPCoordinatorConfig(instance_timeout=0.0)
    with pytest.raises(ValueError):
        MPCoordinatorConfig(health_check_interval=-1.0)


def test_no_env_loading() -> None:
    """Config is CLI-only: the env loader is gone, not merely unused."""
    assert not hasattr(MPCoordinatorConfig, "from_env")


def test_explicit_values_kept() -> None:
    config = MPCoordinatorConfig(
        host="127.0.0.1",
        port=7777,
        instance_timeout=42.0,
        metrics_enabled=False,
        otlp_endpoint="http://collector:4317",
    )
    assert config.host == "127.0.0.1"
    assert config.port == 7777
    assert config.instance_timeout == 42.0
    assert config.metrics_enabled is False
    assert config.otlp_endpoint == "http://collector:4317"
    # Unspecified fields keep their defaults.
    assert config.health_check_interval == MPCoordinatorConfig.health_check_interval
