# SPDX-License-Identifier: Apache-2.0

"""Publishes a ``TIMEOUT_RAISED`` event for each new ``LMCacheTimeoutError``.

The error is defined in :mod:`lmcache.v1.common.timeouts`; ``init_event_bus``
registers :func:`publish_timeout` as its observer.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.timeouts import LMCacheTimeoutError

logger = init_logger(__name__)


def publish_timeout(error: LMCacheTimeoutError, session_id: str) -> None:
    """Publish a ``TIMEOUT_RAISED`` event for ``error``.

    Skipped (no event, no stack capture) when observability is disabled. A
    failure to build or publish the event is logged and swallowed; a failure
    to capture the stack propagates.

    Args:
        error: The timeout being constructed.
        session_id: Request id of the timeout; empty when unavailable.
    """
    if not is_observability_enabled():
        return
    # Stack at the raise site, minus this frame and __init__.
    stacktrace = "".join(traceback.format_stack()[:-2])
    try:
        get_event_bus().publish(
            Event(
                event_type=EventType.TIMEOUT_RAISED,
                metadata={
                    # The constructor argument; a subclass may override __str__.
                    "message": error.args[0],
                    "exception_type": type(error).__name__,
                    "stacktrace": stacktrace,
                },
                session_id=session_id,
            )
        )
    except Exception:
        logger.debug("Failed to publish TIMEOUT_RAISED event", exc_info=True)
