# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.config import (
    HttpCacheEventSourceConfig,
    KafkaCacheEventSourceConfig,
    MPCoordinatorConfig,
)


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


def test_event_source_defaults_to_http() -> None:
    config = MPCoordinatorConfig()
    assert isinstance(config.event_source_config, HttpCacheEventSourceConfig)


def test_kafka_source_config_requires_servers_topic_and_group() -> None:
    with pytest.raises(ValueError, match="bootstrap servers"):
        KafkaCacheEventSourceConfig()
    with pytest.raises(ValueError, match="topic"):
        KafkaCacheEventSourceConfig(bootstrap_servers="broker:9092", topic=" ")
    with pytest.raises(ValueError, match="consumer group"):
        KafkaCacheEventSourceConfig(bootstrap_servers="broker:9092", group_id="")


def test_kafka_source_config_rejects_a_negative_ready_lag() -> None:
    with pytest.raises(ValueError, match="max_ready_lag"):
        KafkaCacheEventSourceConfig(bootstrap_servers="broker:9092", max_ready_lag=-1)


def test_kafka_source_config_defaults() -> None:
    config = KafkaCacheEventSourceConfig(bootstrap_servers="broker:9092")
    assert config.topic == "lmcache-cache-events"
    assert config.group_id == "lmcache-coordinator"
    assert config.max_ready_lag == 1000
