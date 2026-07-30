# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation and env loading."""

# Standard
from typing import Any
from unittest.mock import patch
import os

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def test_defaults_are_valid():
    config = MPCoordinatorConfig()
    assert config.instance_timeout > 0


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
    }
    with patch.dict(os.environ, env, clear=False):
        config = MPCoordinatorConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 7777
    assert config.instance_timeout == 42.0
    # Unset variable keeps the default.
    assert config.health_check_interval == MPCoordinatorConfig.health_check_interval


def test_shared_l1_from_env() -> None:
    env = {
        "LMCACHE_MP_COORDINATOR_SHARED_L1_HOST": "0.0.0.0",
        "LMCACHE_MP_COORDINATOR_SHARED_L1_PORT": "9400",
        "LMCACHE_MP_COORDINATOR_SHARED_L1_AUTHKEY_FILE": (
            "/var/run/secrets/lmcache/shared-l1-authkey"
        ),
        "LMCACHE_MP_COORDINATOR_SHARED_L1_REGION_ID": "cxl-region-0",
        "LMCACHE_MP_COORDINATOR_SHARED_L1_CAPACITY_BYTES": "65536",
        "LMCACHE_MP_COORDINATOR_SHARED_L1_ALIGNMENT_BYTES": "4096",
        "LMCACHE_MP_COORDINATOR_SHARED_L1_LAYOUT_ID": "qwen-layout-v1",
    }
    with patch.dict(os.environ, env, clear=False):
        config = MPCoordinatorConfig.from_env()

    assert config.shared_l1_host == "0.0.0.0"
    assert config.shared_l1_port == 9400
    assert config.shared_l1_authkey_file == "/var/run/secrets/lmcache/shared-l1-authkey"
    assert config.shared_l1_region_id == "cxl-region-0"
    assert config.shared_l1_capacity_bytes == 65536
    assert config.shared_l1_alignment_bytes == 4096
    assert config.shared_l1_layout_id == "qwen-layout-v1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shared_l1_host", "", "shared_l1_host"),
        ("shared_l1_authkey_file", "", "shared_l1_authkey_file"),
        ("shared_l1_region_id", "", "shared_l1_region_id"),
        ("shared_l1_capacity_bytes", 0, "shared_l1_capacity_bytes"),
        ("shared_l1_alignment_bytes", 3, "shared_l1_alignment_bytes"),
        ("shared_l1_layout_id", "", "shared_l1_layout_id"),
    ],
)
def test_enabled_shared_l1_requires_complete_contract(
    field: str,
    value: Any,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "shared_l1_host": "127.0.0.1",
        "shared_l1_port": 9400,
        "shared_l1_authkey_file": "/var/run/secrets/lmcache/shared-l1-authkey",
        "shared_l1_region_id": "region",
        "shared_l1_capacity_bytes": 65536,
        "shared_l1_alignment_bytes": 4096,
        "shared_l1_layout_id": "layout",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        MPCoordinatorConfig(**values)
