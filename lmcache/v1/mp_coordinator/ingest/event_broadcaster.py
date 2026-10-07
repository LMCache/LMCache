# SPDX-License-Identifier: Apache-2.0
"""Fan-out stage of the coordinator's cache-event ingest layer.

The :class:`EventGate` decides *what* reaches the coordinator's state;
this decides *who* sees it. Consumers attach by registration, so adding
one is a wiring change in ``app.py`` and nothing here changes.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.observability import METER_NAME

if TYPE_CHECKING:
    # Third Party
    from opentelemetry.metrics import Meter

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


class CacheEventBroadcaster:
    """Fans one gate-admitted cache-event batch out to every consumer.

    A consumer that raises is logged and counted, and the rest still run,
    so only that consumer misses the batch -- which leaves it disagreeing
    with the others, the failure ``ingest.batch_apply_failures`` exists to
    page on. Keeps no locks: fan-out is thread-safe as long as each
    consumer is.

    Args:
        meter: Meter for the per-consumer instruments. Defaults to the
            global provider's coordinator meter; tests pass a private one.
    """

    def __init__(self, meter: "Meter | None" = None) -> None:
        # Each consumer with its metric attributes, built once at
        # registration rather than on every batch.
        self._consumers: list[tuple[CacheEventConsumer, dict[str, str]]] = []
        if meter is None:
            meter = metrics.get_meter(METER_NAME)
        self._apply_duration = meter.create_histogram(
            "lmcache_coordinator.ingest.batch_apply_duration_seconds",
            unit="s",
            description="Time for one consumer to apply one admitted batch.",
            # In-memory work: sub-millisecond normally, seconds only when
            # something is badly wrong (blend hashing a huge batch).
            explicit_bucket_boundaries_advisory=(
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
            ),
        )
        self._apply_failures = meter.create_counter(
            "lmcache_coordinator.ingest.batch_apply_failures",
            description="Batches or fences a consumer failed to apply; that "
            "consumer now disagrees with the others.",
        )

    def register_consumer(self, consumer: CacheEventConsumer) -> None:
        """Register a consumer for all subsequently broadcast batches.

        Consumers are invoked in registration order. Call during wiring,
        before batches flow: registration is not synchronized against
        concurrent :meth:`broadcast` calls.

        Args:
            consumer: The consumer to fan batches out to.
        """
        self._consumers.append((consumer, {"consumer": type(consumer).__name__}))

    def broadcast(self, batch: CacheEventBatch) -> None:
        """Deliver one gate-admitted batch to every consumer.

        Args:
            batch: The admitted batch.
        """
        for consumer, attributes in self._consumers:
            started = time.perf_counter()
            try:
                consumer.consume(batch)
            except Exception:
                self._apply_failures.add(1, {**attributes, "op": "consume"})
                logger.exception(
                    "Cache-event consumer %s failed on batch %s/%d/%d",
                    type(consumer).__name__,
                    batch.instance_id,
                    batch.incarnation,
                    batch.seq,
                )
            finally:
                self._apply_duration.record(time.perf_counter() - started, attributes)

    def fence_instance(self, instance_id: str) -> None:
        """Tell every consumer that ``instance_id``'s L1 state is void.

        Args:
            instance_id: The restarted or departed instance.
        """
        for consumer, attributes in self._consumers:
            try:
                consumer.fence_instance(instance_id)
            except Exception:
                self._apply_failures.add(1, {**attributes, "op": "fence"})
                logger.exception(
                    "Cache-event consumer %s failed to fence %s",
                    type(consumer).__name__,
                    instance_id,
                )
