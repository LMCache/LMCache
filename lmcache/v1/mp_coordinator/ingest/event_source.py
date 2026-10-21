# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral contract for coordinator cache-event sources.

Sources deliver cache-event batches to :class:`EventGate`. The HTTP source
is the first implementation; a durable source can later use the same gate
without changing admission or consumers.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class EventReplayCapability(str, Enum):
    """Replay capability advertised by a cache-event source."""

    NONE = "none"
    SEEKABLE = "seekable"


@dataclass(frozen=True)
class CacheEventSourceStatus:
    """Status shared by cache-event source implementations.

    Attributes:
        source_name: Human-readable source identity.
        replay_capability: Whether the source can seek retained events.
    """

    source_name: str
    replay_capability: EventReplayCapability


class CacheEventSource(Protocol):
    """Lifecycle and status contract for a cache-event source."""

    async def start(self) -> None:
        """Start source-owned resources."""
        ...

    async def stop(self) -> None:
        """Stop source-owned resources."""
        ...

    def status(self) -> CacheEventSourceStatus:
        """Return the source identity and replay capability.

        Returns:
            The source's current status.
        """
        ...
