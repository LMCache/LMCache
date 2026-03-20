# SPDX-License-Identifier: Apache-2.0

"""Telemetry processor ABCs and config registry."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import TypeVar

# First Party
from lmcache.v1.mp_observability.telemetry.event import TelemetryEvent

T = TypeVar("T", bound="TelemetryProcessorConfig")

# ---------------------------------------------------------------------------
# Registry: processor type name -> config class
# ---------------------------------------------------------------------------

_PROCESSOR_CONFIG_REGISTRY: dict[str, type[TelemetryProcessorConfig]] = {}


def register_telemetry_processor_type(
    name: str, config_cls: type[TelemetryProcessorConfig]
) -> None:
    """Register a telemetry processor config class under a type name.

    Args:
        name: Processor type name (e.g. ``"logging"``).
        config_cls: Config class that can parse from dict via ``from_dict()``.
    """
    if name in _PROCESSOR_CONFIG_REGISTRY:
        raise ValueError(f"Telemetry processor type already registered: {name!r}")
    _PROCESSOR_CONFIG_REGISTRY[name] = config_cls


def get_registered_telemetry_processor_types() -> list[str]:
    """Return the list of registered processor type names."""
    return list(_PROCESSOR_CONFIG_REGISTRY)


# ---------------------------------------------------------------------------
# Base config class for a single processor
# ---------------------------------------------------------------------------


class TelemetryProcessorConfig(ABC):
    """Base class for per-processor configs.

    Each processor type defines a config class that:
    - Subclasses this base.
    - Implements ``from_dict()`` to parse a dict (from JSON) into an instance.
    - Is registered via ``register_telemetry_processor_type()``.
    """

    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Build a config instance from a dict (e.g. from parsed JSON).

        Args:
            d: Processor spec dict (may include type-specific keys).

        Returns:
            An instance of the config class.
        """
        ...

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """Return a help string describing this processor type."""
        ...


# ---------------------------------------------------------------------------
# Processor ABC
# ---------------------------------------------------------------------------


class TelemetryProcessor(ABC):
    """Abstract base class for telemetry event processors."""

    @abstractmethod
    def on_new_event(self, event: TelemetryEvent) -> None:
        """Handle a new telemetry event.

        Called from the drain thread — implementations should be fast.

        Args:
            event: The telemetry event to process.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources when the controller stops."""
        ...
