# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multiprocess config parsing (coordinator registration).

Covers CLI flags, ``LMCACHE_COORDINATOR_*`` env fallback, flag-over-env
precedence, and heartbeat-interval validation. This module imports only the
pure config layer (no native extensions), so it runs without a CUDA build.
"""

# Standard
import argparse
import logging
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    add_coordinator_args,
    add_mp_server_args,
    parse_args_to_coordinator_config,
    parse_args_to_mp_server_config,
)

_COORD_ENV = (
    "LMCACHE_COORDINATOR_URL",
    "LMCACHE_COORDINATOR_ADVERTISE_IP",
    "LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL",
    "LMCACHE_COORDINATOR_EVENT_REPORTING",
    "LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL",
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
    assert config.blend_timeout == 1.0
    assert config.blend_match_concurrency == 8


def test_blend_flags_are_parsed():
    config = _parse(
        [
            "--coordinator-blend-timeout",
            "2.5",
            "--coordinator-blend-match-concurrency",
            "4",
        ]
    )
    assert config.blend_timeout == 2.5
    assert config.blend_match_concurrency == 4


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_blend_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="blend timeout must be a finite number > 0"):
        _parse(["--coordinator-blend-timeout", timeout])


@pytest.mark.parametrize("concurrency", ["0", "-1"])
def test_invalid_blend_concurrency_rejected(concurrency):
    with pytest.raises(ValueError, match="blend match concurrency must be >= 1"):
        _parse(["--coordinator-blend-match-concurrency", concurrency])


def test_blend_flags_have_no_env_fallback(monkeypatch):
    """The blend knobs are CLI-only; the old env names no longer apply."""
    monkeypatch.setenv("LMCACHE_COORDINATOR_BLEND_TIMEOUT", "9.5")
    monkeypatch.setenv("LMCACHE_COORDINATOR_BLEND_MATCH_CONCURRENCY", "32")
    config = _parse([])
    assert config.blend_timeout == 1.0
    assert config.blend_match_concurrency == 8


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


def _parse_mp(argv: list[str]) -> MPServerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parse_args_to_mp_server_config(parser.parse_args(argv))


def test_instance_id_defaults_to_uuid4():
    # No --instance-id flag => a random UUID v4 is minted.
    config = _parse_mp([])
    assert uuid.UUID(config.instance_id).version == 4


def test_instance_id_flag_is_preserved():
    config = _parse_mp(["--instance-id", "mp-server-7"])
    assert config.instance_id == "mp-server-7"


def test_instance_id_defaults_are_distinct():
    # Each parse without the flag gets its own id (no shared default).
    assert _parse_mp([]).instance_id != _parse_mp([]).instance_id


def test_instance_id_dataclass_default_is_distinct():
    # Direct construction (no CLI) also mints a fresh id per instance.
    assert MPServerConfig().instance_id != MPServerConfig().instance_id


# -- Event reporting ----------------------------------------------------------


def test_event_reporting_defaults_are_disabled():
    config = _parse([])
    assert config.event_reporting is False
    assert config.event_flush_interval == 1.0


def test_event_reporting_flags_are_parsed():
    config = _parse(
        [
            "--coordinator-event-reporting",
            "--coordinator-event-flush-interval",
            "0.5",
        ]
    )
    assert config.event_reporting is True
    assert config.event_flush_interval == 0.5


def test_event_reporting_env_fallback(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_EVENT_REPORTING", "true")
    monkeypatch.setenv("LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL", "2.5")
    config = _parse([])
    assert config.event_reporting is True
    assert config.event_flush_interval == 2.5


def test_event_flush_interval_rejects_nonpositive():
    with pytest.raises(ValueError):
        _parse(["--coordinator-event-flush-interval", "0"])


# -- Deprecated pre-v0.5.3 aliases (operator <= v0.5.2 still emits these) -----


def test_deprecated_l2_event_flags_are_accepted():
    config = _parse(
        [
            "--coordinator-l2-event-reporting",
            "--coordinator-l2-event-flush-interval",
            "0.5",
        ]
    )
    assert config.event_reporting is True
    assert config.event_flush_interval == 0.5


def test_new_flags_win_over_deprecated_aliases():
    config = _parse(
        [
            "--coordinator-event-flush-interval",
            "2.0",
            "--coordinator-l2-event-flush-interval",
            "0.5",
        ]
    )
    assert config.event_flush_interval == 2.0


def test_deprecated_flags_log_warning():
    # lmcache's ``init_logger`` sets ``propagate = False``, so pytest's
    # ``caplog`` (root-logger based) cannot see the records. Attach a local
    # handler to the named logger instead (established pattern, see
    # tests/v1/test_v1_adapter_state_desync.py).
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    config_logger = logging.getLogger("lmcache.v1.multiprocess.config")
    config_logger.addHandler(handler)
    try:
        _parse(["--coordinator-l2-event-flush-interval", "0.5"])
    finally:
        config_logger.removeHandler(handler)
    messages = [r.getMessage() for r in records]
    assert any(
        "--coordinator-l2-event-flush-interval is deprecated" in m for m in messages
    )


def test_deprecated_flush_interval_flag_rejects_nonpositive():
    with pytest.raises(ValueError):
        _parse(["--coordinator-l2-event-flush-interval", "0"])
