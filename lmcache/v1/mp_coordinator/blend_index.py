# SPDX-License-Identifier: Apache-2.0
"""Content fingerprint index over the key directory's token bindings.

Answers the blend-style fragment lookup: which cached chunks does a
request's tokens contain, and where? A strided rolling-hash probe
discovers candidates; each is then verified token-exact.

Owns its lock and never calls back into the directory, so the only lock
order is directory → index. Matches name a ``chunk_hash`` only, with no
model or salt awareness.

See ``docs/design/v1/mp_coordinator/blend_index.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import BlendMatch
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
)

logger = init_logger(__name__)

# Fleet-constant polynomial base. Both sides of the match live here, so
# this never has to agree with anything a server computes; it is fixed so
# stored and probed fingerprints align within a coordinator's lifetime.
POLY_BASE = np.uint64(0x9E3779B97F4A7C15)

# Filter size = smallest power of two >= _TABLE_GROWTH * live contents.
# The load factor is the filter's false-positive rate (a wasted dict
# lookup on a query position), so it is kept low — at one byte per slot
# this costs ~16 bytes per indexed chunk.
_TABLE_GROWTH = 16
_MIN_TABLE_SIZE = 1 << 10


@dataclass(frozen=True)
class BlendIndexStats:
    """A point-in-time summary of index contents.

    Attributes:
        num_contents: Distinct chunk contents indexed.
        num_chunks: Chunks indexed across those contents.
        table_size: Slots in the occupancy filter.
    """

    num_contents: int
    num_chunks: int
    table_size: int


@dataclass
class _FingerprintEntry:
    """One distinct content: the tokens to verify against, plus every
    chunk holding it. ``token_ids`` is the first indexed chunk's content,
    so a fingerprint collision with different content stays
    undiscoverable rather than being wrongly matched."""

    token_ids: np.ndarray
    # (chunk_hash, token_offset); the offset is per chunk, since identical
    # content can be stored at different positions.
    occupants: list[tuple[bytes, int]] = field(default_factory=list)


class BlendIndex:
    """Thread-safe content fingerprint index for fragment lookups.

    Mutations arrive through :meth:`add` and :meth:`remove` (driven by
    binding lifecycle); reads through :meth:`match` and :meth:`stats`.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 1) -> None:
        """Initialize an empty index.

        Args:
            chunk_size: Tokens per indexed chunk, and the match window.
                Content of any other length is not indexable.
            probe_stride: Query positions between probes; ``1`` gives
                full recall.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is < 1.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
        if probe_stride < 1:
            raise ValueError(f"probe_stride must be >= 1 (got {probe_stride})")
        self._chunk_size = chunk_size
        self._probe_stride = probe_stride
        self._lock = threading.Lock()
        # Exact resolution: fingerprint -> content. Authoritative.
        self._fingerprint_table: dict[int, _FingerprintEntry] = {}
        # Occupancy filter: 1 where some fingerprint lands. Deliberately
        # carries no identity, so a bucket shared by two fingerprints
        # simply admits both to the dict lookup instead of hiding one.
        self._slots = np.zeros(_MIN_TABLE_SIZE, dtype=np.uint8)
        self._mask = np.uint64(_MIN_TABLE_SIZE - 1)
        # Bits currently set; a removal leaves its bit behind (a stale bit
        # only costs a dict miss), so rebuild once they outgrow the entries.
        self._bits_set = 0

    def add(self, token_ids: np.ndarray, chunk_hash: bytes, token_offset: int) -> None:
        """Index ``chunk_hash``'s content, or attach it to an existing entry.

        Idempotent. Content whose length is not ``chunk_size`` is ignored.

        Args:
            token_ids: The chunk's tokens. Held by reference, so the
                caller must not mutate it afterwards.
            chunk_hash: The chunk's ``ObjectKey.chunk_hash``.
            token_offset: The chunk's position in its stored sequence.
        """
        if token_ids.shape[0] != self._chunk_size:
            logger.warning(
                "Not indexing chunk content of %d tokens: the index matches "
                "a %d-token window, so only full-chunk content is "
                "discoverable (fleet chunk-size disagreement?)",
                token_ids.shape[0],
                self._chunk_size,
            )
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._fingerprint_table.get(poly)
            if entry is None:
                self._fingerprint_table[poly] = _FingerprintEntry(
                    token_ids=token_ids, occupants=[(chunk_hash, token_offset)]
                )
                slot = poly & int(self._mask)
                if not self._slots[slot]:
                    self._slots[slot] = 1
                    self._bits_set += 1
                if _TABLE_GROWTH * len(self._fingerprint_table) > self._slots.shape[0]:
                    self._rebuild_table()
                return
            for index, (held_hash, _) in enumerate(entry.occupants):
                if held_hash == chunk_hash:
                    entry.occupants[index] = (chunk_hash, token_offset)
                    return
            entry.occupants.append((chunk_hash, token_offset))

    def remove(self, token_ids: np.ndarray, chunk_hash: bytes) -> None:
        """Drop ``chunk_hash`` from the entry for ``token_ids``.

        The content is dropped with its last chunk; removing an unknown
        chunk or content is a no-op.

        Args:
            token_ids: The content the chunk was indexed under.
            chunk_hash: The chunk to drop.
        """
        if token_ids.shape[0] != self._chunk_size:
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._fingerprint_table.get(poly)
            if entry is None:
                return
            entry.occupants = [
                occupant for occupant in entry.occupants if occupant[0] != chunk_hash
            ]
            if entry.occupants:
                return
            del self._fingerprint_table[poly]
            # The bit stays until a rebuild: clearing it here could hide a
            # different fingerprint sharing the bucket.
            if self._bits_set > 2 * len(self._fingerprint_table):
                self._rebuild_table()

    def match(self, tokens: np.ndarray) -> list[BlendMatch]:
        """Find indexed chunks contained in ``tokens``.

        Every candidate is verified token-exact before acceptance.

        Args:
            tokens: The query token ids (any dtype castable to
                ``uint64``).

        Returns:
            Matches in ascending ``cur_st`` order, at most one per chunk.
        """
        query = np.asarray(tokens, dtype=np.uint64)
        if query.shape[0] < self._chunk_size:
            return []
        rolling = rolling_hash_windows_numba(query, self._chunk_size, POLY_BASE)
        probe = rolling[:: self._probe_stride]
        window = self._chunk_size
        matches: list[BlendMatch] = []
        seen: set[bytes] = set()
        with self._lock:
            # One gather through the occupancy filter, then exact dict
            # resolution on the survivors. A shared bucket admits both
            # fingerprints, so recall stays complete.
            occupied = self._slots[probe & self._mask]
            for position in np.nonzero(occupied)[0].tolist():
                entry = self._fingerprint_table.get(int(probe[position]))
                if entry is None:
                    continue  # bucket shared with another fingerprint
                cur_st = position * self._probe_stride
                if not np.array_equal(query[cur_st : cur_st + window], entry.token_ids):
                    continue  # fingerprint collision: content differs
                for chunk_hash, token_offset in entry.occupants:
                    if chunk_hash in seen:
                        continue
                    seen.add(chunk_hash)
                    matches.append(
                        BlendMatch(
                            chunk_hash=chunk_hash,
                            old_st=token_offset,
                            cur_st=cur_st,
                        )
                    )
                    break  # occupants are content-identical; one suffices
        return matches

    def stats(self) -> BlendIndexStats:
        """Return a point-in-time summary of index contents.

        Returns:
            Distinct contents, total chunks, and the table size.
        """
        with self._lock:
            return BlendIndexStats(
                num_contents=len(self._fingerprint_table),
                num_chunks=sum(
                    len(entry.occupants) for entry in self._fingerprint_table.values()
                ),
                table_size=int(self._slots.shape[0]),
            )

    # -- Internals -------------------------------------------------------------

    def _fingerprint(self, token_ids: np.ndarray) -> int:
        """Return the 64-bit polynomial fingerprint of one chunk's content."""
        window = np.asarray(token_ids, dtype=np.uint64)
        return int(chunk_hash_windows_numba(window, self._chunk_size, POLY_BASE)[0])

    def _rebuild_table(self) -> None:
        """Resize the occupancy filter and rebuild it from live contents,
        clearing bits left behind by removals. Call with the lock held."""
        size = _MIN_TABLE_SIZE
        while size < _TABLE_GROWTH * len(self._fingerprint_table):
            size <<= 1
        self._slots = np.zeros(size, dtype=np.uint8)
        self._mask = np.uint64(size - 1)
        if self._fingerprint_table:
            polys = np.fromiter(
                self._fingerprint_table.keys(),
                dtype=np.uint64,
                count=len(self._fingerprint_table),
            )
            self._slots[polys & self._mask] = 1
        self._bits_set = int(np.count_nonzero(self._slots))
