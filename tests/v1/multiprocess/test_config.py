# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multiprocess config parsing (coordinator registration).

Covers CLI flags, ``LMCACHE_COORDINATOR_*`` env fallback, flag-over-env
precedence, and heartbeat-interval validation. This module imports only the
pure config layer (no native extensions), so it runs without a CUDA build.
"""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    add_coordinator_args,
    parse_args_to_coordinator_config,
)

_COORD_ENV = (
    "LMCACHE_COORDINATOR_URL",
    "LMCACHE_COORDINATOR_ADVERTISE_IP",
    "LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL",
)


def _parse(argv: list[str]) -> CoordinatorConfig:
    parser = argparse.ArgumentParser()
    add_coordinator_args(parser)
    return parse_args_to_coordinator_config(parser.parse_args(argv))


@pytest.fixture(autouse=True)
def _clear_coord_env(monkeypatch):
    """Isolate each test from any coordinator env vars in the environment."""
    for name in _COORD_ENV:
        monkeypatch.delenv(name, raising=False)


def test_defaults_disable_registration():
    config = _parse([])
    assert config.url == ""  # empty url => registration disabled
    assert config.advertise_ip == ""
    assert config.heartbeat_interval == 5.0


def test_flags_are_parsed():
    config = _parse(
        [
            "--coordinator-url",
            "http://coord:9300",
            "--coordinator-advertise-ip",
            "10.0.0.5",
            "--coordinator-heartbeat-interval",
            "2.5",
        ]
    )
    assert config.url == "http://coord:9300"
    assert config.advertise_ip == "10.0.0.5"
    assert config.heartbeat_interval == 2.5


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_URL", "http://env-coord:9300")
    monkeypatch.setenv("LMCACHE_COORDINATOR_ADVERTISE_IP", "192.168.1.2")
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", "3")
    config = _parse([])
    assert config.url == "http://env-coord:9300"
    assert config.advertise_ip == "192.168.1.2"
    assert config.heartbeat_interval == 3.0


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_URL", "http://env-coord:9300")
    config = _parse(["--coordinator-url", "http://flag-coord:9300"])
    assert config.url == "http://flag-coord:9300"


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_heartbeat_rejected(interval):
    # Non-positive and non-finite (nan/inf) values are all rejected.
    with pytest.raises(ValueError, match="finite number > 0"):
        _parse(["--coordinator-heartbeat-interval", interval])


@pytest.mark.parametrize("interval", ["nan", "inf"])
def test_invalid_heartbeat_from_env_rejected(monkeypatch, interval):
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", interval)
    with pytest.raises(ValueError, match="finite number > 0"):
        _parse([])


def test_garbage_env_heartbeat_rejected(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", "abc")
    with pytest.raises(ValueError, match="not a number"):
        _parse([])
