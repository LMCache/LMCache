# SPDX-License-Identifier: Apache-2.0
"""Whether the coordinator's view is current enough to act on.

A coordinator whose ingest source is still catching up on a durable
stream knows about part of the fleet's cache and none of the rest.
Reading that view is harmless -- a lookup that misses is a miss.
*Planning* against it is not: quotas are metadata, restored in full
before ingest starts, so a controller that plans while lag is still high
compares a restored limit against a fraction of the actual usage and
orders evictions the fleet does not need.

The coordinator's answer to that is to not be a coordinator yet: the
lifespan blocks on :meth:`IngestReadiness.wait_until_ready` before
starting any controller (see ``app.py``), so there is no partially-ready
state anything else has to reason about -- a controller that got to
``run`` at all is running against a caught-up view.

Exactly one source feeds the gate (see ``app.py``), so this wraps that
one source rather than a collection of them.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from dataclasses import dataclass
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.ingest.event_source import (
    UNKNOWN_LAG,
    CacheEventSource,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class IngestReadinessStatus:
    """The readiness answer, with what it was derived from.

    Attributes:
        ready: Whether the source has caught up enough to act on.
        lag: Batches the source is behind by, or :data:`UNKNOWN_LAG`
            (from :mod:`.event_source`) when it cannot say.
        max_lag: The budget ``lag`` is compared against.
    """

    ready: bool
    lag: int
    max_lag: int


class IngestReadiness:
    """Compares the ingest source's lag against an operator's budget.

    A source that cannot say how far behind it is counts as not ready:
    that is the safe direction, since a coordinator that has lost sight
    of the stream must not start acting on the last view it had, and
    holding costs nothing but a delayed startup.

    The HTTP source never lags (it applies each request as it arrives),
    so a deployment that has not configured a durable transport is
    always ready -- startup behaves exactly as it did before this existed.

    Args:
        source: The one source feeding the gate.
        max_lag: Batches the coordinator may be behind and still act.
    """

    def __init__(self, source: CacheEventSource, max_lag: int) -> None:
        self._source = source
        self._max_lag = max_lag

    def status(self) -> IngestReadinessStatus:
        """Return the current readiness answer.

        Returns:
            The verdict, the source's lag, and the budget it was
            compared against.
        """
        lag = self._source.status().lag
        if lag == UNKNOWN_LAG:
            return IngestReadinessStatus(
                ready=False, lag=UNKNOWN_LAG, max_lag=self._max_lag
            )
        return IngestReadinessStatus(
            ready=lag <= self._max_lag, lag=lag, max_lag=self._max_lag
        )

    async def wait_until_ready(
        self, poll_interval: float = 0.5, log_interval: float = 10.0
    ) -> None:
        """Block until the source has caught up enough to act on.

        Returns immediately for a source that is already within budget --
        an HTTP deployment always is, since its lag is always ``0``. No
        timeout: a source that never catches up (an unreachable broker)
        holds the coordinator here rather than let it start with a view
        it knows is wrong, and the periodic log line is how that shows up
        as more than a coordinator that never finished starting.

        Args:
            poll_interval: Seconds between readiness checks.
            log_interval: Minimum seconds between progress log lines, so
                a long catch-up does not flood the log.
        """
        last_logged = float("-inf")
        while True:
            current = self.status()
            if current.ready:
                return
            now = time.monotonic()
            if now - last_logged >= log_interval:
                logger.info(
                    "Waiting for ingest to catch up: lag=%s (budget %d)",
                    current.lag,
                    current.max_lag,
                )
                last_logged = now
            await asyncio.sleep(poll_interval)
