# SPDX-License-Identifier: Apache-2.0
"""Server-side timeout telemetry; exception identity is client-owned."""

# Standard
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.multiprocess.client_errors import (
    LMCacheTimeoutError,
    register_timeout_observer,
)

__all__ = ["LMCacheTimeoutError", "observe_timeout"]

logger = init_logger(__name__)


def observe_timeout(error: LMCacheTimeoutError, session_id: str) -> None:
    """Publish a ``TIMEOUT_RAISED`` event for ``error``; swallow any failure.

    Skipped when observability is disabled.

    Args:
        error: The timeout being constructed.
        session_id: Request id of the timeout; empty when unavailable.
    """
    if is_observability_enabled():
        try:
            get_event_bus().publish(
                Event(
                    event_type=EventType.TIMEOUT_RAISED,
                    metadata={
                        "message": str(error),
                        "exception_type": type(error).__name__,
                        # The raise site, without this frame and __init__.
                        "stacktrace": "".join(traceback.format_stack()[:-2]),
                    },
                    session_id=session_id,
                )
            )
        except Exception:
            logger.debug("Failed to publish TIMEOUT_RAISED event", exc_info=True)


register_timeout_observer(observe_timeout)
