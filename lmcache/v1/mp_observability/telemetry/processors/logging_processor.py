# SPDX-License-Identifier: Apache-2.0

"""Built-in logging telemetry processor."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.telemetry.event import TelemetryEvent
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
    TelemetryProcessorConfig,
    register_telemetry_processor_type,
)

logger = init_logger(__name__)


class LoggingProcessorConfig(TelemetryProcessorConfig):
    """Config for the built-in logging processor (no extra fields)."""

    def __init__(self, log_level: str) -> None:
        self.log_level = log_level

    @classmethod
    def from_dict(cls, d: dict) -> LoggingProcessorConfig:
        log_level = d.get("log_level", "DEBUG").upper()
        return cls(log_level=log_level)

    @classmethod
    def help(cls) -> str:
        return (
            "Logging processor: logs telemetry events at specified level.\n"
            "Config fields:\n"
            "  log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
            "Default is DEBUG."
        )


register_telemetry_processor_type("logging", LoggingProcessorConfig)


class LoggingProcessor(TelemetryProcessor):
    """Processor that logs telemetry events at DEBUG level."""

    def __init__(self, config: LoggingProcessorConfig) -> None:
        self.log_func = getattr(logger, config.log_level.lower(), logger.debug)

    def on_new_event(self, event: TelemetryEvent) -> None:
        self.log_func(
            "Telemetry: %s %s session=%s ts=%.6f metadata=%s",
            event.name,
            event.event_type.value,
            event.session_id,
            event.timestamp,
            event.metadata,
        )

    def shutdown(self) -> None:
        pass
