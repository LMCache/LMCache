# SPDX-License-Identifier: Apache-2.0
"""Fan-out stage of the coordinator's cache-event ingest layer.

The :class:`EventGate` decides *what* reaches the coordinator's state;
this decides *who* sees it. Consumers attach by registration, so adding
one is a wiring change in ``app.py`` and nothing here changes.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from typing import Protocol

# First Party
from lmcache.v1.mp_coordinator.api import CacheEventBatch


class CacheEventConsumer(Protocol):
    """One downstream consumer of gate-admitted cache-event batches."""

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one gate-admitted batch to this consumer's state.

        Called once per admitted batch, in admission order. Only
        per-instance ordering is guaranteed, and skipping irrelevant
        tiers and event types is the consumer's own job.

        Args:
            batch: The admitted batch.
        """
        ...

    def fence_instance(self, instance_id: str) -> None:
        """Discard the **L1** state ``instance_id`` reported, before any
        batch of its new incarnation is consumed.

        Called on a restart or a departure. L2 bytes outlive the
        reporting process, so L2-only consumers no-op.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """
        ...


class CacheEventBroadcaster:
    """Fans one gate-admitted cache-event batch out to every consumer.

    Keeps no locks: fan-out is thread-safe as long as each consumer is.
    """

    def __init__(self) -> None:
        self._consumers: list[CacheEventConsumer] = []

    def register_consumer(self, consumer: CacheEventConsumer) -> None:
        """Register a consumer for all subsequently broadcast batches.

        Consumers are invoked in registration order. Call during wiring,
        before batches flow: registration is not synchronized against
        concurrent :meth:`broadcast` calls.

        Args:
            consumer: The consumer to fan batches out to.
        """
        self._consumers.append(consumer)

    def broadcast(self, batch: CacheEventBatch) -> None:
        """Deliver one gate-admitted batch to every consumer.

        Args:
            batch: The admitted batch.
        """
        for consumer in self._consumers:
            consumer.consume(batch)

    def fence_instance(self, instance_id: str) -> None:
        """Tell every consumer that ``instance_id``'s L1 state is void.

        Args:
            instance_id: The restarted or departed instance.
        """
        for consumer in self._consumers:
            consumer.fence_instance(instance_id)
