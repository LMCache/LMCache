# SPDX-License-Identifier: Apache-2.0

"""Tests for LoggingProcessor and LoggingProcessorConfig."""

# Standard
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)
from lmcache.v1.mp_observability.telemetry.processors import logging_processor
from lmcache.v1.mp_observability.telemetry.processors.logging_processor import (
    LoggingProcessor,
    LoggingProcessorConfig,
)

_LOGGER_NAME = logging_processor.__name__


@pytest.fixture(autouse=True)
def _enable_propagation():
    """Temporarily enable propagation so caplog can capture."""
    lg = logging.getLogger(_LOGGER_NAME)
    old = lg.propagate
    lg.propagate = True
    yield
    lg.propagate = old


class TestLoggingProcessor:
    def test_on_new_event_logs_at_debug(self, caplog):
        proc = LoggingProcessor(LoggingProcessorConfig.from_dict({}))
        event = TelemetryEvent(
            name="mp.store",
            event_type=EventType.START,
            session_id="sess-1",
        )

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            proc.on_new_event(event)

        assert "Telemetry:" in caplog.text
        assert "mp.store" in caplog.text
        assert "START" in caplog.text
        assert "sess-1" in caplog.text

    def test_metadata_appears_in_log(self, caplog):
        proc = LoggingProcessor(LoggingProcessorConfig.from_dict({}))
        event = TelemetryEvent(
            name="mp.lookup",
            event_type=EventType.END,
            session_id="sess-2",
            metadata={"tokens": 512},
        )

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            proc.on_new_event(event)

        assert "512" in caplog.text

    def test_custom_log_level(self, caplog):
        proc = LoggingProcessor(LoggingProcessorConfig.from_dict({"log_level": "INFO"}))
        event = TelemetryEvent(
            name="mp.store",
            event_type=EventType.START,
            session_id="sess-3",
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            proc.on_new_event(event)

        assert "Telemetry:" in caplog.text
        assert "mp.store" in caplog.text

    def test_shutdown_does_not_raise(self):
        proc = LoggingProcessor(LoggingProcessorConfig.from_dict({}))
        proc.shutdown()


class TestLoggingProcessorConfig:
    def test_from_dict_empty(self):
        config = LoggingProcessorConfig.from_dict({})
        assert isinstance(config, LoggingProcessorConfig)
        assert config.log_level == "DEBUG"

    def test_from_dict_custom_log_level(self):
        config = LoggingProcessorConfig.from_dict({"log_level": "info"})
        assert config.log_level == "INFO"

    def test_help_returns_string(self):
        h = LoggingProcessorConfig.help()
        assert isinstance(h, str)
        assert len(h) > 0
