# SPDX-License-Identifier: Apache-2.0
"""Routes applied cache-event batches into the coordinator's consumers.

The key directory is the ordering/dedup gate for the fleet's single
cache-event stream — and, since its L2 view survives reporter restarts,
also the L2 usage ledger. Batches it *applies* are fanned out here to
every registered :class:`CacheEventConsumer` (today: the eviction LRU).
Any future ingestion path (e.g. a message-queue consumer) calls the same
router, so consumers are independent of the transport — and the router
is independent of the consumers: they attach through
:meth:`CacheEventBroadcaster.register_consumer` (the same registration shape
as ``EventBus.register_subscriber``), so adding one is a wiring change
in ``app.py``, not a router change.
"""

# Standard
from typing import Protocol

# First Party
from lmcache.v1.mp_coordinator.api import CacheEventBatch


class CacheEventConsumer(Protocol):
    """One downstream consumer of directory-applied cache-event batches."""

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one directory-applied batch to this consumer's state.

        Called once per applied batch, in application order. The batch
        has already passed the directory's dedup/fencing gate, so a
        consumer sees each event at most once per delivery attempt and
        must not assume more than per-instance ordering. Events the
        consumer does not care about (tiers, event types) are its own
        job to skip.

        Args:
            batch: The applied batch.
        """
        ...


class CacheEventBroadcaster:
    """Fans one applied cache-event batch out to every registered consumer.

    Consumers attach via :meth:`register_consumer` after construction
    (mirroring ``EventBus.register_subscriber``); the router takes no
    dependency on any of them. The router itself keeps no locks: routing
    is thread-safe as long as each consumer's ``consume`` is.
    """

    def __init__(self) -> None:
        self._consumers: list[CacheEventConsumer] = []

    def register_consumer(self, consumer: CacheEventConsumer) -> None:
        """Register a consumer for all subsequently routed batches.

        Consumers are invoked in registration order. Call during wiring,
        before batches flow (``create_app`` does); registration is not
        synchronized against concurrent :meth:`route` calls.

        Args:
            consumer: The consumer to fan batches out to.
        """
        self._consumers.append(consumer)

    def broadcast(self, batch: CacheEventBatch) -> None:
        """Deliver one batch to every consumer.

        Call only for batches the key directory applied — replays and
        stale incarnations are already dropped there.

        Args:
            batch: The applied batch.
        """
        for consumer in self._consumers:
            consumer.consume(batch)
