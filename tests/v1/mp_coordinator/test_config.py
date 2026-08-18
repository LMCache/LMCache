# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation and env loading."""

# Standard
from unittest.mock import patch
import os

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


def test_from_env_overrides_and_falls_back():
    env = {
        "LMCACHE_MP_COORDINATOR_HOST": "127.0.0.1",
        "LMCACHE_MP_COORDINATOR_PORT": "7777",
        "LMCACHE_MP_COORDINATOR_INSTANCE_TIMEOUT": "42",
        "LMCACHE_MP_COORDINATOR_METRICS_ENABLED": "false",
        "LMCACHE_MP_COORDINATOR_OTLP_ENDPOINT": "http://collector:4317",
    }
    with patch.dict(os.environ, env, clear=False):
        config = MPCoordinatorConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 7777
    assert config.instance_timeout == 42.0
    assert config.metrics_enabled is False
    assert config.otlp_endpoint == "http://collector:4317"
    # Unset variable keeps the default.
    assert config.health_check_interval == MPCoordinatorConfig.health_check_interval


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_metrics_enabled_truthy_env_values(value: str) -> None:
    with patch.dict(
        os.environ,
        {"LMCACHE_MP_COORDINATOR_METRICS_ENABLED": value},
        clear=True,
    ):
        config = MPCoordinatorConfig.from_env()
    assert config.metrics_enabled is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "invalid"])
def test_metrics_enabled_falsy_env_values(value: str) -> None:
    with patch.dict(
        os.environ,
        {"LMCACHE_MP_COORDINATOR_METRICS_ENABLED": value},
        clear=True,
    ):
        config = MPCoordinatorConfig.from_env()
    assert config.metrics_enabled is False
