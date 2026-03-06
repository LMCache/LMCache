# SPDX-License-Identifier: Apache-2.0

"""Telemetry event data model."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Type of telemetry event: START or END of a traced operation."""

    START = "START"
    END = "END"


@dataclass
class TelemetryEvent:
    """A single telemetry event for tracing operations across async boundaries.

    Attributes:
        name: Operation name, e.g. ``"mp.store"``, ``"mp.lookup"``.
        event_type: Whether this is the START or END of the operation.
        session_id: Caller-provided ID that correlates START/END pairs.
        metadata: Flat key-value pairs compatible with OTel attributes.
        timestamp: Wall-clock time (``time.time()``) set by
            ``TelemetryController.log()`` when the event is ingested.
            Defaults to ``0.0`` at construction; call sites should NOT
            set this — it is stamped automatically.
    """

    name: str
    event_type: EventType
    session_id: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    timestamp: float = 0.0
