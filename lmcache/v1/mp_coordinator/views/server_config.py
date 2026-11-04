# SPDX-License-Identifier: Apache-2.0
"""Per-MP-server memory capacity, declared by ``config`` cache events.

Cache events report bytes held, never bytes holdable, so servers declare
capacity separately and this registry stores it. It is a
:class:`~lmcache.v1.mp_coordinator.ingest.event_broadcaster.CacheEventConsumer`
like the key directory and the usage manager, so declarations arrive through
the same gate -- inheriting its incarnation fencing, dedup, and ordering
instead of carrying a second mechanism alongside.

Compartments are :class:`ModuleMemoryCapacity`, the same type the MP server
builds them as -- one shape for one concept on both sides of the wire.

One declaration is one ``config`` batch per compartment, all sharing a
``capacity_revision``. A batch whose revision is newer than what is stored
starts a fresh set; batches at the same revision extend it. That is what
retires a compartment: a declaration that omits it simply never adds it.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import cast
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ModuleMemoryCapacity, Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)

# "No cap declared". Real caps are positive, so an unlimited adapter
# reporting 0 would otherwise read as permanently full.
UNDECLARED_CAPACITY = 0


class ServerConfigRegistry(View):
    """Thread-safe store of each MP server's declared capacities.

    A :class:`CacheEventConsumer`: :meth:`consume` accumulates ``config``
    batches into one declaration per ``(incarnation, capacity_revision)``,
    so a compartment the newest declaration omits stops being reported.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # Every HTTP caller is an async handler on one event loop, so the
        # lock is not for them. It is for :meth:`capture`: a quiesced
        # capture blocks on a condition variable waiting for the in-flight
        # batch, which only works off the ingest thread -- so a capture
        # genuinely races :meth:`consume`. Unsynchronized, it would iterate
        # ``_stamps`` while ``consume`` reassigns ``_by_instance``.
        self._lock = threading.Lock()
        self._by_instance: dict[str, dict[tuple[Tier, str], ModuleMemoryCapacity]] = {}
        self._stamps: dict[str, tuple[int, int]] = {}

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one ``config`` batch; ignore every other event type.

        The gate has already dropped stale incarnations and duplicates, so
        the only ordering left to do is grouping batches into declarations:
        a newer ``(incarnation, capacity_revision)`` starts a fresh set, an
        equal one extends it, an older one is a straggler and is dropped.

        Args:
            batch: A gate-admitted batch. Non-``config`` batches no-op.
        """
        if batch.event_type != CacheEventType.CONFIG:
            return
        module = ModuleMemoryCapacity(
            tier=batch.tier,
            backend=batch.backend,
            capacity_bytes=batch.capacity_bytes,
            shared=batch.shared,
        )
        stamp = (batch.incarnation, batch.capacity_revision)
        with self._lock:
            stored = self._stamps.get(batch.instance_id, (-1, -1))
            if stamp < stored:
                return
            if stamp > stored:
                # A new declaration: drop the previous set so compartments
                # it no longer lists stop being reported.
                self._by_instance[batch.instance_id] = {}
                self._stamps[batch.instance_id] = stamp
            self._by_instance[batch.instance_id][(module.tier, module.backend)] = module

    def fence_instance(self, instance_id: str) -> None:
        """No-op: capacity is configuration, not reported L1 state.

        A restarting process redeclares under a higher incarnation, which
        :meth:`consume` supersedes on its own. A departing one has its
        declaration dropped by :meth:`forget`.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """

    def get(self, instance_id: str) -> tuple[ModuleMemoryCapacity, ...]:
        """Return ``instance_id``'s declared compartments.

        Args:
            instance_id: The server to look up.

        Returns:
            Its declarations; empty when unknown or nothing was declared.
        """
        with self._lock:
            return tuple(self._by_instance.get(instance_id, {}).values())

    def get_all(self) -> dict[str, tuple[ModuleMemoryCapacity, ...]]:
        """Return a snapshot of every server's declarations.

        Returns:
            A copy mapping ``instance_id`` to its compartments.
        """
        with self._lock:
            return {
                instance_id: tuple(modules.values())
                for instance_id, modules in self._by_instance.items()
            }

    def forget(self, instance_id: str) -> None:
        """Drop ``instance_id``'s declaration. Idempotent.

        Args:
            instance_id: The departed server.
        """
        with self._lock:
            self._by_instance.pop(instance_id, None)
            self._stamps.pop(instance_id, None)

    def get_durable_components(self) -> tuple[DurableComponent, ...]:
        """Return this registry: the capacities it holds are its section.

        Returns:
            Itself.
        """
        return (self,)

    @property
    def name(self) -> str:
        """Name of the capacity registry's section in a checkpoint."""
        return "server_config"

    @property
    def persistence_type(self) -> PersistenceType:
        """Declarations come off the cache-event stream and every server
        redeclares on registration, so they are checkpoint state."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return each server's declaration with the stamp that ordered it.

        The stamp travels with the modules. Without it a restored registry
        would start from scratch and accept a straggler from before the
        capture, regressing the topology it just loaded.

        Returns:
            ``{"declarations": [(instance_id, incarnation, revision,
            [(tier, backend, capacity_bytes, shared), ...]), ...]}``.
        """
        with self._lock:
            return {
                "declarations": [
                    (
                        instance_id,
                        incarnation,
                        revision,
                        [
                            (
                                module.tier.value,
                                module.backend,
                                module.capacity_bytes,
                                module.shared,
                            )
                            for module in self._by_instance.get(
                                instance_id, {}
                            ).values()
                        ],
                    )
                    for instance_id, (incarnation, revision) in self._stamps.items()
                ]
            }

    def restore(self, state: Mapping[str, object]) -> None:
        """Load captured declarations and the stamps that ordered them.

        Call once at startup.

        Args:
            state: A :meth:`capture` value.

        Raises:
            ValueError: If any declaration is already held.
        """
        declarations = cast(
            "list[tuple[str, int, int, list[tuple[str, str, int, bool]]]]",
            state["declarations"],
        )
        with self._lock:
            if self._by_instance or self._stamps:
                raise ValueError(
                    "restore() requires an empty registry (holds "
                    f"{len(self._by_instance)} declarations)"
                )
            for instance_id, incarnation, revision, modules in declarations:
                self._stamps[instance_id] = (incarnation, revision)
                self._by_instance[instance_id] = {
                    (Tier(tier_value), backend): ModuleMemoryCapacity(
                        tier=Tier(tier_value),
                        backend=backend,
                        capacity_bytes=capacity_bytes,
                        shared=shared,
                    )
                    for tier_value, backend, capacity_bytes, shared in modules
                }
