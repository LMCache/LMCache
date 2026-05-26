# SPDX-License-Identifier: Apache-2.0
"""Protocol and types for pluggable engine modules."""

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Protocol

# First Party
from lmcache.v1.multiprocess.protocol import RequestType


class ThreadPoolType(Enum):
    """Declares which thread pool a handler should run in."""

    SYNC = auto()
    AFFINITY = auto()
    NORMAL = auto()


@dataclass
class HandlerSpec:
    """Specification for a single message queue handler.

    Args:
        request_type: The ZMQ request type this handler serves.
        handler: The callable that processes the request.
        pool: Which thread pool the handler runs in.
    """

    request_type: RequestType
    handler: Callable
    pool: ThreadPoolType


class EngineModule(Protocol):
    """Protocol for pluggable engine modules.

    Each module owns its internal state and exposes handlers
    that the compositor registers with the message queue server.
    """

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves."""
        ...

    def report_status(self) -> dict:
        """Return module-specific status information."""
        ...

    def close(self) -> None:
        """Release resources owned by this module."""
        ...
