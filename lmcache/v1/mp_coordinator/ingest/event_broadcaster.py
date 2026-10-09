# SPDX-License-Identifier: Apache-2.0
"""Fan-out stage of the coordinator's cache-event ingest layer.

The :class:`EventGate` decides *what* reaches the coordinator's state;
this decides *who* sees it. Consumers attach by registration, so adding
one is a wiring change in ``app.py`` and nothing here changes.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import CacheEventBatch

logger = init_logger(__name__)


@runtime_checkable
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


@dataclass(frozen=True)
class ConsumerStats:
    """What one consumer has done with the batches handed to it since this
    process started.

    Attributes:
        batches_delivered: Batches handed to its ``consume``, failed or not.
        apply_seconds: Total time spent in those ``consume`` calls.
        consume_failures: ``consume`` calls that raised; each leaves this
            consumer disagreeing with the others.
        fence_failures: ``fence_instance`` calls that raised.
    """

    batches_delivered: int = 0
    apply_seconds: float = 0.0
    consume_failures: int = 0
    fence_failures: int = 0


@dataclass
class _ConsumerTally:
    """Mutable form of :class:`ConsumerStats`."""

    batches_delivered: int = 0
    apply_seconds: float = 0.0
    consume_failures: int = 0
    fence_failures: int = 0


class CacheEventBroadcaster:
    """Fans one gate-admitted cache-event batch out to every consumer.

    A consumer that raises is logged and counted, and the rest still run,
    so only that consumer misses the batch -- which leaves it disagreeing
    with the others. Fan-out takes no lock, so it is thread-safe as long as
    each consumer is; a small lock guards only the per-consumer tallies
    behind :meth:`stats`.
    """

    def __init__(self) -> None:
        self._consumers: list[tuple[CacheEventConsumer, _ConsumerTally]] = []
        self._tally_lock = threading.Lock()

    def register_consumer(self, consumer: CacheEventConsumer) -> None:
        """Register a consumer for all subsequently broadcast batches.

        Consumers are invoked in registration order. Call during wiring,
        before batches flow: registration is not synchronized against
        concurrent :meth:`broadcast` calls.

        Args:
            consumer: The consumer to fan batches out to.
        """
        self._consumers.append((consumer, _ConsumerTally()))

    def broadcast(self, batch: CacheEventBatch) -> None:
        """Deliver one gate-admitted batch to every consumer.

        Args:
            batch: The admitted batch.
        """
        for consumer, tally in self._consumers:
            started = time.perf_counter()
            failed = False
            try:
                consumer.consume(batch)
            except Exception:
                failed = True
                logger.exception(
                    "Cache-event consumer %s failed on batch %s/%d/%d",
                    type(consumer).__name__,
                    batch.instance_id,
                    batch.incarnation,
                    batch.seq,
                )
            elapsed = time.perf_counter() - started
            with self._tally_lock:
                tally.batches_delivered += 1
                tally.apply_seconds += elapsed
                if failed:
                    tally.consume_failures += 1

    def fence_instance(self, instance_id: str) -> None:
        """Tell every consumer that ``instance_id``'s L1 state is void.

        Args:
            instance_id: The restarted or departed instance.
        """
        for consumer, tally in self._consumers:
            try:
                consumer.fence_instance(instance_id)
            except Exception:
                with self._tally_lock:
                    tally.fence_failures += 1
                logger.exception(
                    "Cache-event consumer %s failed to fence %s",
                    type(consumer).__name__,
                    instance_id,
                )

    def stats(self) -> dict[str, ConsumerStats]:
        """Return each consumer's tally since this process started.

        Returns:
            :class:`ConsumerStats` keyed by the consumer's class name.
            Discovery builds one instance per class, so the names are
            unique.
        """
        with self._tally_lock:
            return {
                type(consumer).__name__: ConsumerStats(
                    batches_delivered=tally.batches_delivered,
                    apply_seconds=tally.apply_seconds,
                    consume_failures=tally.consume_failures,
                    fence_failures=tally.fence_failures,
                )
                for consumer, tally in self._consumers
            }
