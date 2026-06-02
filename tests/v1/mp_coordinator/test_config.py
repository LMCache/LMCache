# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation and env loading."""

# Standard
from unittest.mock import patch
import os

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def test_defaults_are_valid():
    config = MPCoordinatorConfig()
    assert config.instance_timeout > config.heartbeat_interval


def test_instance_timeout_must_exceed_heartbeat_interval():
    with pytest.raises(ValueError):
        MPCoordinatorConfig(heartbeat_interval=5.0, instance_timeout=5.0)


def test_non_positive_intervals_rejected():
    with pytest.raises(ValueError):
        MPCoordinatorConfig(heartbeat_interval=0.0)
    with pytest.raises(ValueError):
        MPCoordinatorConfig(instance_timeout=0.0)
    with pytest.raises(ValueError):
        MPCoordinatorConfig(health_check_interval=-1.0)


def test_from_env_overrides_and_falls_back():
    env = {
        "LMCACHE_MP_COORDINATOR_REPLY_URL": "127.0.0.1:7777",
        "LMCACHE_MP_COORDINATOR_INSTANCE_TIMEOUT": "42",
    }
    with patch.dict(os.environ, env, clear=False):
        config = MPCoordinatorConfig.from_env()
    assert config.reply_url == "127.0.0.1:7777"
    assert config.instance_timeout == 42.0
    # Unset variable keeps the default.
    assert config.pull_url == MPCoordinatorConfig.pull_url
