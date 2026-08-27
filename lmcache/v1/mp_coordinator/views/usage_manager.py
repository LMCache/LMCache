# SPDX-License-Identifier: Apache-2.0
"""Fleet cache usage view for the MP coordinator, across both tiers.

Byte totals derived from the same admitted cache-event stream that
builds the key directory, rolled up two ways: per ``cache_salt`` (the
tenant axis, what eviction enforces quotas against) and per
``(instance_id, backend)`` (the capacity axis, how full one node's L1
is). A ``CacheEventConsumer`` in its own right — no single controller
owns it, because eviction reads only the L2 half.

L1 and L2 differ in what removes bytes. L2 bytes outlive their reporter
(they sit on storage the fleet shares, so only a ``DELETE`` event
removes them), while L1 bytes are the reporter's own process memory and
are void the moment it restarts or leaves. The ingest gate detects both
and calls :meth:`fence_instance`.

See ``docs/design/v1/mp_coordinator/usage_and_eviction.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import cast
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.utils.encoding import decode_key, encode_key
from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)

# One tracked placement: ``(tier, key, owner, backend)``, the identity the
# key directory upserts on. ``owner`` is the reporting instance, or ``""``
# for a shared pool — fleet-scoped bytes that outlive any one member, so
# no owner's fencing removes them.
_PlacementId = tuple[Tier, ObjectKey, str, str]


class CacheUsageManager(View):
    """Thread-safe per-tier byte usage view, rolled up two ways.

    Every read is tier-explicit: a key resident in both tiers holds
    bytes in both, so a tier-blind total would double-count it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._placement_sizes: dict[_PlacementId, int] = {}
        self._key_bytes: dict[tuple[Tier, ObjectKey], int] = {}
        self._salt_bytes: dict[tuple[Tier, str], int] = {}
        self._instance_bytes: dict[tuple[Tier, str, str], int] = {}
        # L1 placements by owning instance, so fencing one costs its own
        # placements rather than a full scan.
        self._l1_by_instance: dict[str, set[_PlacementId]] = {}

    def consume(self, batch: CacheEventBatch) -> None:
        """Account one gate-admitted batch: ``STORE`` upserts placement
        bytes (delta on re-store), ``DELETE`` removes them.

        Args:
            batch: The admitted batch; ``ACCESS`` carries no placement
                identity and is ignored.
        """
        if batch.event_type == CacheEventType.ACCESS:
            return
        owner = "" if batch.shared else batch.instance_id
        with self._lock:
            for entry in batch.entries:
                key = entry.key.to_object_key()
                placement_id = (batch.tier, key, owner, batch.backend)
                if batch.event_type == CacheEventType.STORE:
                    old = self._placement_sizes.get(placement_id, 0)
                    self._placement_sizes[placement_id] = entry.size_bytes
                    if batch.tier == Tier.L1 and owner:
                        self._l1_by_instance.setdefault(owner, set()).add(placement_id)
                    self._adjust(placement_id, entry.size_bytes - old)
                elif batch.event_type == CacheEventType.DELETE:
                    self._remove(placement_id)

    def fence_instance(self, instance_id: str) -> None:
        """Drop every L1 byte ``instance_id`` reported.

        Those bytes were the reporting process's memory and died with
        it. Its L2 bytes stay: they outlive the reporter and leave only
        via ``DELETE`` events. Shared pools are owned by the fleet, not
        by any one reporter, so they are untouched.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """
        with self._lock:
            # Popped first, so the index cleanup in _remove is a no-op and
            # cannot mutate the set being iterated.
            for placement_id in self._l1_by_instance.pop(instance_id, set()):
                self._remove(placement_id)

    def get_salt_bytes(self, tier: Tier, cache_salt: str) -> int:
        """Return ``tier`` bytes currently held under ``cache_salt``."""
        with self._lock:
            return self._salt_bytes.get((tier, cache_salt), 0)

    def get_bytes_by_salt(self, tier: Tier) -> dict[str, int]:
        """Return a snapshot of ``tier`` bytes per ``cache_salt``."""
        with self._lock:
            return {
                salt: n_bytes
                for (placement_tier, salt), n_bytes in self._salt_bytes.items()
                if placement_tier == tier
            }

    def get_bytes_by_instance(self, tier: Tier) -> dict[str, dict[str, int]]:
        """Return a snapshot of ``tier`` bytes per instance, per backend.

        Keyed ``instance_id`` → ``backend`` → bytes. Shared pools are not
        owned by any one instance and appear under the ``""`` key.
        """
        with self._lock:
            by_instance: dict[str, dict[str, int]] = {}
            for (
                placement_tier,
                instance_id,
                backend,
            ), n_bytes in self._instance_bytes.items():
                if placement_tier == tier:
                    by_instance.setdefault(instance_id, {})[backend] = n_bytes
            return by_instance

    def get_key_bytes(self, tier: Tier, key: ObjectKey) -> int:
        """Return the ``tier`` bytes held for ``key`` across all its
        tracked placements (``0`` when it has none in that tier)."""
        with self._lock:
            return self._key_bytes.get((tier, key), 0)

    def get_total_bytes(self, tier: Tier) -> int:
        """Return total ``tier`` bytes tracked across all salts."""
        with self._lock:
            return sum(
                n_bytes
                for (placement_tier, _), n_bytes in self._salt_bytes.items()
                if placement_tier == tier
            )

    def get_durable_components(self) -> tuple[DurableComponent, ...]:
        """Return this view: the bytes it accounts are its own section.

        Returns:
            Itself, since nothing else owns the placement sizes.
        """
        return (self,)

    @property
    def name(self) -> str:
        """Name of the usage view's section in a checkpoint."""
        return "cache_usage"

    @property
    def persistence_type(self) -> PersistenceType:
        """Byte accounting follows the placements it came from, so it is
        checkpoint state."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return the per-placement byte sizes.

        Only the placements: every read this class serves is a rollup of
        them, so persisting the rollups too would store the same bytes
        twice and admit totals that disagree with their placements.

        Returns:
            ``{"placements": [(tier, key, owner, backend, size), ...]}``.
        """
        with self._lock:
            return {
                "placements": [
                    (tier.value, encode_key(key), owner, backend, size_bytes)
                    for (tier, key, owner, backend), size_bytes in (
                        self._placement_sizes.items()
                    )
                ]
            }

    def restore(self, state: Mapping[str, object]) -> None:
        """Load captured placements, rebuilding every rollup from them.

        Call once at startup.

        Args:
            state: A :meth:`capture` value.

        Raises:
            ValueError: If any bytes are already accounted.
        """
        placements = cast(
            "list[tuple[str, object, str, str, int]]", state["placements"]
        )
        with self._lock:
            if self._placement_sizes:
                raise ValueError(
                    "restore() requires an empty usage view (holds "
                    f"{len(self._placement_sizes)} placements)"
                )
            for tier_value, encoded_key, owner, backend, size_bytes in placements:
                tier = Tier(tier_value)
                placement_id = (tier, decode_key(encoded_key), owner, backend)
                self._placement_sizes[placement_id] = size_bytes
                if tier == Tier.L1 and owner:
                    self._l1_by_instance.setdefault(owner, set()).add(placement_id)
                self._adjust(placement_id, size_bytes)

    # -- Internals (call with self._lock held) --------------------------------

    def _remove(self, placement_id: _PlacementId) -> None:
        """Drop ``placement_id`` from tracking and subtract its bytes."""
        tier, _, owner, _ = placement_id
        owned = self._l1_by_instance.get(owner) if tier == Tier.L1 else None
        if owned is not None:
            owned.discard(placement_id)
            if not owned:
                del self._l1_by_instance[owner]
        self._adjust(placement_id, -self._placement_sizes.pop(placement_id, 0))

    def _adjust(self, placement_id: _PlacementId, delta: int) -> None:
        """Apply ``delta`` bytes to every rollup ``placement_id`` feeds.

        A rollup that falls to zero is dropped rather than kept at zero,
        so the snapshot reads carry only live entries and the maps do not
        grow with every key the fleet has ever evicted.
        """
        if delta == 0:
            return
        tier, key, owner, backend = placement_id

        key_id = (tier, key)
        key_total = self._key_bytes.get(key_id, 0) + delta
        if key_total > 0:
            self._key_bytes[key_id] = key_total
        else:
            self._key_bytes.pop(key_id, None)

        salt_id = (tier, key.cache_salt)
        salt_total = self._salt_bytes.get(salt_id, 0) + delta
        if salt_total > 0:
            self._salt_bytes[salt_id] = salt_total
        else:
            self._salt_bytes.pop(salt_id, None)
            if salt_total < 0:
                logger.warning(
                    "%s usage underflow for cache_salt=%r (delta %d); clamping to 0",
                    tier.value,
                    key.cache_salt,
                    delta,
                )

        node_id = (tier, owner, backend)
        node_total = self._instance_bytes.get(node_id, 0) + delta
        if node_total > 0:
            self._instance_bytes[node_id] = node_total
        else:
            self._instance_bytes.pop(node_id, None)
