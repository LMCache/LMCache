# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide key directory for the MP coordinator.

Maps each :class:`ObjectKey` to its placements (instance, tier, backend,
size) across the fleet, built by consuming the cache-event stream the
ingest layer has already ordered, deduped, and fenced. Eventually
consistent: lookups are hints to be validated at the owner. L1
placements die with their reporter's incarnation; L2 placements persist.

See ``docs/design/v1/mp_coordinator/views/key_directory.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    UNKNOWN_TOKEN_OFFSET,
    BlendMatch,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.blend_index import BlendIndex, BlendIndexStats
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.views.base import View

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry

# First Party
from lmcache.v1.mp_coordinator.utils.encoding import decode_key, encode_key

logger = init_logger(__name__)

# Token ids are held as ``uint32``: a few hundred bytes per chunk instead
# of the ~10 KB a ``tuple[int, ...]`` of boxed ints costs, and content
# comparison against a query window stays vectorized.
_TOKEN_DTYPE = np.uint32

_CAPTURED_TOKEN_DTYPE = np.dtype("<u4")
"""Token ids are captured little-endian regardless of host order, so a
capture written on one machine restores on another."""

# Shared empty array for chunks whose content is unknown.
_NO_TOKENS = np.empty(0, dtype=_TOKEN_DTYPE)


@dataclass(frozen=True)
class Placement:
    """One live placement of a key, as returned by directory lookups.

    Attributes:
        instance_id: The emitter that most recently reported the placement.
        incarnation: The reporting instance's incarnation at report time.
        tier: Tier the bytes live on (``l1`` or ``l2``).
        backend: Backend within the tier.
        size_bytes: Size the owner reported at store time.
        shared: ``True`` when the backend is a fleet-shared pool (see
            :class:`CacheEventBatch`).
    """

    instance_id: str
    incarnation: int
    tier: Tier
    backend: str
    size_bytes: int
    shared: bool = False


@dataclass(frozen=True)
class DirectoryStats:
    """A point-in-time summary of directory contents.

    Attributes:
        num_keys: Keys with at least one placement.
        num_placements: Total placements across all keys.
        l1_keys_by_instance: Keys each instance reported L1 placements
            for; its stream cursor lives on the ingest gate.
        blend: How much of the directory is fragment-matchable.
    """

    num_keys: int
    num_placements: int
    l1_keys_by_instance: dict[str, int]
    blend: BlendIndexStats


@dataclass
class _KeyRecord:
    """Directory value for one key: its placements plus recency."""

    placements: list[Placement] = field(default_factory=list)
    last_access: float = 0.0


@dataclass
class _TokenBinding:
    """Token content for one chunk hash plus the keys sharing it (dropped
    when the last key goes). ``token_ids`` is empty until a
    token-bearing ``STORE`` entry arrives."""

    token_ids: np.ndarray
    token_offset: int
    keys: set[ObjectKey]


class KeyDirectory(View):
    """Thread-safe in-memory key directory built from cache events.

    Mutations arrive through the ingest layer's two consumer hooks,
    :meth:`consume` and :meth:`fence_instance`; reads through
    :meth:`lookup` and :meth:`stats`. Its contents survive a
    coordinator restart through :meth:`capture` and :meth:`restore`.

    Fragment (blend) lookup is off until :meth:`enable_blend_lookup` is
    called, so a fleet that does not run CacheBlend hashes no content.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._directory: dict[ObjectKey, _KeyRecord] = {}
        # instance_id → keys it reported L1 placements for. The reverse
        # index that makes fencing proportional to the instance's own
        # keys instead of a full directory scan.
        self._l1_keys_by_instance: dict[str, set[ObjectKey]] = {}
        # chunk hash → tokens + keys, for chunk hashes of >= 1 record.
        self._token_bindings: dict[bytes, _TokenBinding] = {}
        # Derived from the bindings; owns its own lock (order: self → index).
        # A real (1 KiB) index rather than None keeps blend_stats() branch-free;
        # enable_blend_lookup() swaps in one sized to the fleet's chunk size.
        self._blend_index = BlendIndex()
        self._blend_lookup_enabled = False

    @classmethod
    def from_config(
        cls, config: "MPCoordinatorConfig", views: "Registry[View]"
    ) -> "KeyDirectory":
        """Build the directory, indexing chunk content only if asked.

        Blend lookup lives here rather than at the call site because the
        window it matches on is the fleet chunk size -- the same value the
        directory keys on.

        Args:
            config: The coordinator configuration.
            views: Unused; the directory depends on nothing.
        """
        directory = cls()
        if config.enable_blend_lookup:
            directory.enable_blend_lookup(
                chunk_size=config.chunk_size, probe_stride=config.blend_probe_stride
            )
        return directory

    def get_durable_components(self) -> tuple[DurableComponent, ...]:
        """Return this directory: nothing else knows what the fleet cached.

        Returns:
            Itself.
        """
        return (self,)

    def enable_blend_lookup(self, chunk_size: int, probe_stride: int) -> None:
        """Start indexing chunk content so :meth:`blend_match` can serve.

        Call once at startup. Chunks stored before the call are not
        retroactively indexed.

        Args:
            chunk_size: The index's match window; must equal the MP
                servers' chunk size.
            probe_stride: Query positions between probes.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is < 1.
        """
        self._blend_index = BlendIndex(chunk_size=chunk_size, probe_stride=probe_stride)
        self._blend_lookup_enabled = True

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one gate-admitted batch, idempotently: re-storing
        upserts the placement (and its token binding), deleting an
        absent placement is a no-op.

        Args:
            batch: The admitted batch.
        """
        with self._lock:
            l1_keys = self._l1_keys_by_instance.setdefault(batch.instance_id, set())
            for entry in batch.entries:
                self._apply_entry(l1_keys, batch, entry)

    def lookup(self, keys: list[ObjectKey]) -> list[list[Placement]]:
        """Return the known placements for each requested key.

        Args:
            keys: The keys to look up.

        Returns:
            One placement list per requested key, in request order —
            empty for unknown keys. Each list is sorted by
            ``(instance_id, tier, backend)``.
        """
        with self._lock:
            results: list[list[Placement]] = []
            for key in keys:
                record = self._directory.get(key)
                if record is None:
                    results.append([])
                    continue
                results.append(
                    sorted(
                        record.placements,
                        key=lambda p: (p.instance_id, p.tier.value, p.backend),
                    )
                )
            return results

    def get_token_ids(self, chunk_hashes: list[bytes]) -> list[tuple[int, ...]]:
        """Return the known token ids for each requested chunk hash.

        Args:
            chunk_hashes: ``ObjectKey.chunk_hash`` values to look up.

        Returns:
            One token-id tuple per hash, in request order — empty for
            unknown chunks.
        """
        with self._lock:
            return [
                tuple(binding.token_ids.tolist())
                if (binding := self._token_bindings.get(chunk_hash)) is not None
                else ()
                for chunk_hash in chunk_hashes
            ]

    def blend_match(self, tokens: np.ndarray) -> list[BlendMatch]:
        """Find cached chunks contained anywhere in ``tokens``.

        Unlike :meth:`lookup` the query need not be a prefix. Matches name
        a ``chunk_hash`` only, which the caller expands with its own
        model, salt, and world size. Takes the blend index's lock, not
        the directory's.

        Args:
            tokens: The query token ids.

        Returns:
            Matches in ascending ``cur_st`` order, at most one per chunk;
            empty when :meth:`enable_blend_lookup` was never called. They
            may overlap in the query, so callers that scatter them must
            resolve overlaps themselves.
        """
        if not self._blend_lookup_enabled:
            return []
        return self._blend_index.match(tokens)

    def blend_stats(self) -> BlendIndexStats:
        """Return a point-in-time summary of the blend index.

        Returns:
            Distinct contents, total chunks, and the filter size.
        """
        return self._blend_index.stats()

    def list_keys(
        self,
        tier: Tier = Tier.ALL,
        instance_id: str = "",
        backend: str = "",
        offset: int = 0,
        limit: int = 1000,
    ) -> tuple[int, dict[ObjectKey, list[Placement]]]:
        """List keys whose placements match the filters, one page at a time.

        A snapshot for inspection: iteration order is the directory's
        insertion order and is not stable across mutations, so pages of
        a changing directory may skip or repeat keys.

        Args:
            tier: Keep placements on this tier (``all`` keeps every tier).
            instance_id: Keep placements reported by this instance
                (empty keeps every instance).
            backend: Keep placements on this backend (empty keeps every
                backend).
            offset: Matching keys to skip.
            limit: Maximum keys to return.

        Returns:
            ``(total, page)``: the number of keys with at least one
            matching placement, and the ``[offset, offset + limit)``
            slice of them as an ordered mapping of key → its matching
            placements.

        Raises:
            ValueError: If ``offset`` or ``limit`` is negative.
        """
        if offset < 0:
            raise ValueError(f"offset must be >= 0 (got {offset})")
        if limit < 0:
            raise ValueError(f"limit must be >= 0 (got {limit})")
        with self._lock:
            total = 0
            page: dict[ObjectKey, list[Placement]] = {}
            for key, record in self._directory.items():
                placements = [
                    p
                    for p in record.placements
                    if (tier == Tier.ALL or p.tier == tier)
                    and (not instance_id or p.instance_id == instance_id)
                    and (not backend or p.backend == backend)
                ]
                if not placements:
                    continue
                if total >= offset and len(page) < limit:
                    page[key] = placements
                total += 1
            return total, page

    def fence_instance(self, instance_id: str) -> None:
        """Remove every **L1** placement reported by ``instance_id``.

        L2 placements survive: their bytes persist across the reporter's
        restarts and leave only via ``DELETE`` events.

        Args:
            instance_id: The instance whose L1 placements to drop.
        """
        with self._lock:
            removed = self._drop_l1_placements(instance_id)
            self._l1_keys_by_instance.pop(instance_id, None)
        if removed:
            logger.info(
                "Fenced instance %s: dropped %d L1 placement(s)",
                instance_id,
                removed,
            )

    @property
    def name(self) -> str:
        """Name of the directory's section in a checkpoint."""
        return "key_directory"

    @property
    def persistence_type(self) -> PersistenceType:
        """Placements come from the event stream, so they are checkpoint
        state: losing them costs hit rate, not correctness."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return the directory's contents.

        Returns:
            ``{"keys": [(key, last_access, [placement, ...]), ...],
            "bindings": [(chunk_hash, token_offset, token bytes), ...],
            "l1_keys_by_instance": {instance_id: [key index, ...]}}``.
            The blend index is derived, so it is absent.
        """
        with self._lock:
            keys = [
                (
                    encode_key(key),
                    record.last_access,
                    [_encode_placement(p) for p in record.placements],
                )
                for key, record in self._directory.items()
            ]
            positions = {key: index for index, key in enumerate(self._directory)}
            return {
                "keys": keys,
                "bindings": [
                    (
                        chunk_hash,
                        binding.token_offset,
                        binding.token_ids.astype(
                            _CAPTURED_TOKEN_DTYPE, copy=False
                        ).tobytes(),
                    )
                    for chunk_hash, binding in self._token_bindings.items()
                ],
                "l1_keys_by_instance": {
                    instance_id: [positions[key] for key in l1_keys if key in positions]
                    for instance_id, l1_keys in self._l1_keys_by_instance.items()
                },
            }

    def restore(self, state: Mapping[str, object]) -> None:
        """Load a captured directory into an empty one.

        Call once at startup. Blend fingerprints are re-derived, so
        :meth:`enable_blend_lookup` must run first for restored chunks to
        be fragment-matchable.

        Args:
            state: A :meth:`capture` value.

        Raises:
            ValueError: If the directory already holds keys, or the
                reverse index cites a key position that does not exist.
        """
        captured_keys = cast("list[tuple[object, float, list[object]]]", state["keys"])
        bindings = cast("list[tuple[bytes, int, bytes]]", state["bindings"])
        l1_by_instance = cast("Mapping[str, list[int]]", state["l1_keys_by_instance"])
        with self._lock:
            if self._directory or self._l1_keys_by_instance:
                raise ValueError(
                    "restore() requires an empty directory (holds "
                    f"{len(self._directory)} keys, "
                    f"{len(self._l1_keys_by_instance)} instances)"
                )
            key_table = []
            for encoded_key, last_access, placements in captured_keys:
                key = decode_key(encoded_key)
                key_table.append(key)
                self._directory[key] = _KeyRecord(
                    placements=[_decode_placement(p) for p in placements],
                    last_access=last_access,
                )
                self._add_token_binding(key)
            for chunk_hash, token_offset, raw_tokens in bindings:
                binding = self._token_bindings.get(chunk_hash)
                if binding is None:
                    # Content no surviving key references: nothing owns it.
                    continue
                token_ids = np.frombuffer(
                    raw_tokens, dtype=_CAPTURED_TOKEN_DTYPE
                ).astype(_TOKEN_DTYPE, copy=False)
                token_ids.flags.writeable = False
                binding.token_ids = token_ids
                binding.token_offset = token_offset
                if self._blend_lookup_enabled and token_offset != UNKNOWN_TOKEN_OFFSET:
                    self._blend_index.add(token_ids, chunk_hash, token_offset)
            for instance_id, positions_ in l1_by_instance.items():
                for position in positions_:
                    if not 0 <= position < len(key_table):
                        raise ValueError(
                            f"{instance_id!r} cites key position {position} "
                            f"of {len(key_table)}"
                        )
                self._l1_keys_by_instance[instance_id] = {
                    key_table[position] for position in positions_
                }

    def stats(self) -> DirectoryStats:
        """Return a point-in-time summary of directory contents."""
        blend = self._blend_index.stats()
        with self._lock:
            num_placements = sum(
                len(record.placements) for record in self._directory.values()
            )
            return DirectoryStats(
                num_keys=len(self._directory),
                num_placements=num_placements,
                l1_keys_by_instance={
                    instance_id: len(keys)
                    for instance_id, keys in self._l1_keys_by_instance.items()
                },
                blend=blend,
            )

    # -- Internals (call with self._lock held) --------------------------------

    def _apply_entry(
        self,
        l1_keys: set[ObjectKey],
        batch: CacheEventBatch,
        entry: CacheEventEntry,
    ) -> None:
        """Apply one entry under the directory lock, maintaining
        ``l1_keys`` (the emitter's L1 reverse index)."""
        key = entry.key.to_object_key()
        if batch.event_type == CacheEventType.STORE:
            record = self._directory.get(key)
            if record is None:
                record = _KeyRecord()
                self._directory[key] = record
                self._add_token_binding(key)
            placement = Placement(
                instance_id=batch.instance_id,
                incarnation=batch.incarnation,
                tier=batch.tier,
                backend=batch.backend,
                size_bytes=entry.size_bytes,
                shared=batch.shared,
            )
            index = self._find_placement(record.placements, batch)
            if index is None:
                record.placements.append(placement)
            else:
                record.placements[index] = placement
            if entry.token_ids:
                self._create_token_binding(key.chunk_hash, entry)
            record.last_access = max(record.last_access, batch.ts)
            if batch.tier == Tier.L1:
                l1_keys.add(key)
        elif batch.event_type == CacheEventType.DELETE:
            record = self._directory.get(key)
            if record is None:
                return
            index = self._find_placement(record.placements, batch)
            if index is not None:
                record.placements.pop(index)
            if not record.placements:
                del self._directory[key]
                self._remove_token_binding(key)
            if batch.tier == Tier.L1 and not any(
                p.tier == Tier.L1 and p.instance_id == batch.instance_id
                for p in record.placements
            ):
                l1_keys.discard(key)
        elif batch.event_type == CacheEventType.ACCESS:
            record = self._directory.get(key)
            if record is not None:
                record.last_access = max(record.last_access, batch.ts)

    @staticmethod
    def _find_placement(
        placements: list[Placement], batch: CacheEventBatch
    ) -> int | None:
        """Return the index of the placement whose identity matches
        ``batch``, or ``None`` if absent."""
        for index, placement in enumerate(placements):
            if (
                placement.shared == batch.shared
                and (batch.shared or placement.instance_id == batch.instance_id)
                and placement.tier == batch.tier
                and placement.backend == batch.backend
            ):
                return index
        return None

    def _drop_l1_placements(self, instance_id: str) -> int:
        """Remove and count the **L1** placements ``instance_id`` reported."""
        l1_keys = self._l1_keys_by_instance.get(instance_id)
        if l1_keys is None:
            return 0
        removed = 0
        for key in l1_keys:
            record = self._directory.get(key)
            if record is None:
                continue
            kept = [
                p
                for p in record.placements
                if p.tier != Tier.L1 or p.instance_id != instance_id
            ]
            removed += len(record.placements) - len(kept)
            if kept:
                record.placements = kept
            else:
                del self._directory[key]
                self._remove_token_binding(key)
        l1_keys.clear()
        return removed

    def _create_token_binding(self, chunk_hash: bytes, entry: CacheEventEntry) -> None:
        """Record ``entry``'s token content on ``chunk_hash``'s binding.

        Token ids outside ``uint32`` leave the binding as it was, so one
        bad entry is a lookup miss rather than a failed batch. An entry
        whose ``token_offset`` is
        :data:`~lmcache.v1.mp_coordinator.api.UNKNOWN_TOKEN_OFFSET` fills
        the binding's content but is not indexed: a fragment match with
        no stored position would re-RoPE from the wrong source.

        Args:
            chunk_hash: Chunk hash whose binding to fill.
            entry: The store entry carrying the token ids and offset.
        """
        try:
            token_ids = np.asarray(entry.token_ids, dtype=_TOKEN_DTYPE)
        except (OverflowError, TypeError, ValueError):
            logger.warning(
                "Ignoring token ids for chunk %s: values outside uint32",
                chunk_hash.hex(),
            )
            return
        token_ids.flags.writeable = False
        binding = self._token_bindings[chunk_hash]
        if (
            self._blend_lookup_enabled
            and binding.token_ids.size
            and not np.array_equal(binding.token_ids, token_ids)
        ):
            # Re-store with different content: retire the old fingerprint,
            # or the chunk stays discoverable under content it no longer has.
            self._blend_index.remove(binding.token_ids, chunk_hash)
        binding.token_ids = token_ids
        binding.token_offset = entry.token_offset
        if not self._blend_lookup_enabled:
            return
        if entry.token_offset == UNKNOWN_TOKEN_OFFSET:
            return
        self._blend_index.add(token_ids, chunk_hash, entry.token_offset)

    def _add_token_binding(self, key: ObjectKey) -> None:
        """Index ``key`` under its chunk's token binding, creating an
        empty binding on first reference."""
        binding = self._token_bindings.get(key.chunk_hash)
        if binding is None:
            self._token_bindings[key.chunk_hash] = _TokenBinding(
                token_ids=_NO_TOKENS, token_offset=UNKNOWN_TOKEN_OFFSET, keys={key}
            )
        else:
            binding.keys.add(key)

    def _remove_token_binding(self, key: ObjectKey) -> None:
        """Remove ``key`` from its chunk's token binding, dropping the
        binding — and its blend-index entry — with its last key."""
        binding = self._token_bindings.get(key.chunk_hash)
        if binding is None:
            return
        binding.keys.discard(key)
        if not binding.keys:
            del self._token_bindings[key.chunk_hash]
            if self._blend_lookup_enabled and binding.token_ids.size:
                self._blend_index.remove(binding.token_ids, key.chunk_hash)


# -- Durable encoding ---------------------------------------------------------
#
# Positional tuples rather than mappings: a capture runs on the caller's
# thread over a fleet's worth of keys, and per-key field names cost more
# than the fields themselves.


def _encode_placement(placement: Placement) -> tuple[str, int, str, str, int, bool]:
    """Flatten a placement for a capture."""
    return (
        placement.instance_id,
        placement.incarnation,
        placement.tier.value,
        placement.backend,
        placement.size_bytes,
        placement.shared,
    )


def _decode_placement(encoded: object) -> Placement:
    """Rebuild a placement from :func:`_encode_placement`."""
    instance_id, incarnation, tier, backend, size_bytes, shared = cast(
        "tuple[str, int, str, str, int, bool]", encoded
    )
    return Placement(
        instance_id=instance_id,
        incarnation=incarnation,
        tier=Tier(tier),
        backend=backend,
        size_bytes=size_bytes,
        shared=shared,
    )
