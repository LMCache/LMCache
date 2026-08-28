# SPDX-License-Identifier: Apache-2.0
"""Content fingerprint index over the key directory's token bindings.

Answers the blend-style fragment lookup: which cached chunks does a
request's tokens contain, and where? A strided rolling-hash probe
discovers candidates; each is then verified token-exact.

One content can be held by several chunk hashes (the same text stored
after different prefixes), each claimed by the namespaces that stored it;
a match names one the requester's own namespace claims.

Owns its lock and never calls back into the directory, so the only lock
order is directory → index.

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
from lmcache.v1.mp_coordinator.api import BlendMatch, BlendNamespace
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
        num_claims: ``(chunk, namespace)`` pairs across those chunks.
        num_namespaces: Distinct namespaces holding indexed content.
        table_size: Slots in the occupancy filter.
    """

    num_contents: int
    num_chunks: int
    num_claims: int
    num_namespaces: int
    table_size: int


@dataclass
class _Occupant:
    """One chunk hash holding an entry's content.

    Attributes:
        token_offset: The chunk's position in its stored sequence; per
            chunk, since identical content can sit at different offsets.
        namespaces: Every namespace that stored this chunk. The occupant
            is dropped when the last one goes.
    """

    token_offset: int
    namespaces: set[BlendNamespace] = field(default_factory=set)


@dataclass
class _FingerprintEntry:
    """One distinct content: the tokens to verify against, plus every
    chunk holding it. ``token_ids`` is the first indexed chunk's content,
    so a fingerprint collision with different content stays
    undiscoverable rather than being wrongly matched."""

    token_ids: np.ndarray
    # chunk_hash -> occupant, in first-indexed order.
    occupants: dict[bytes, _Occupant] = field(default_factory=dict)


class BlendIndex:
    """Thread-safe content fingerprint index for fragment lookups.

    Mutations arrive through :meth:`add`, :meth:`remove`, and
    :meth:`remove_chunk` (driven by binding lifecycle); reads through
    :meth:`match` and :meth:`stats`.
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

    def add(
        self,
        token_ids: np.ndarray,
        chunk_hash: bytes,
        token_offset: int,
        namespace: BlendNamespace,
    ) -> None:
        """Record that ``namespace`` stores ``chunk_hash`` with this content.

        Idempotent, and additive in the namespace: a second namespace
        joins the existing occupant rather than replacing its claim.
        Content whose length is not ``chunk_size`` is ignored.

        Args:
            token_ids: The chunk's tokens. Held by reference, so the
                caller must not mutate it afterwards.
            chunk_hash: The chunk's ``ObjectKey.chunk_hash``.
            token_offset: The chunk's position in its stored sequence.
            namespace: The namespace the chunk was stored in.
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
                entry = _FingerprintEntry(token_ids=token_ids)
                self._fingerprint_table[poly] = entry
                slot = poly & int(self._mask)
                if not self._slots[slot]:
                    self._slots[slot] = 1
                    self._bits_set += 1
                if _TABLE_GROWTH * len(self._fingerprint_table) > self._slots.shape[0]:
                    self._rebuild_table()
            occupant = entry.occupants.get(chunk_hash)
            if occupant is None:
                entry.occupants[chunk_hash] = _Occupant(
                    token_offset=token_offset, namespaces={namespace}
                )
                return
            occupant.token_offset = token_offset
            occupant.namespaces.add(namespace)

    def remove(
        self, token_ids: np.ndarray, chunk_hash: bytes, namespace: BlendNamespace
    ) -> None:
        """Drop ``namespace``'s claim on ``chunk_hash``.

        The chunk survives while another namespace still stores it, and
        the content is dropped with its last chunk. Removing an unknown
        claim, chunk, or content is a no-op.

        Args:
            token_ids: The content the chunk was indexed under.
            chunk_hash: The chunk to release.
            namespace: The namespace releasing it.
        """
        if token_ids.shape[0] != self._chunk_size:
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._fingerprint_table.get(poly)
            if entry is None:
                return
            occupant = entry.occupants.get(chunk_hash)
            if occupant is None:
                return
            occupant.namespaces.discard(namespace)
            if occupant.namespaces:
                return
            del entry.occupants[chunk_hash]
            self._drop_entry_if_empty(poly, entry)

    def remove_chunk(self, token_ids: np.ndarray, chunk_hash: bytes) -> None:
        """Drop ``chunk_hash`` and every namespace's claim on it.

        For a re-store under the same hash with different tokens, which
        invalidates every claim at once.

        Args:
            token_ids: The content the chunk was indexed under.
            chunk_hash: The chunk to drop.
        """
        if token_ids.shape[0] != self._chunk_size:
            return
        poly = self._fingerprint(token_ids)
        with self._lock:
            entry = self._fingerprint_table.get(poly)
            if entry is None or entry.occupants.pop(chunk_hash, None) is None:
                return
            self._drop_entry_if_empty(poly, entry)

    def match(self, tokens: np.ndarray, namespace: BlendNamespace) -> list[BlendMatch]:
        """Find chunks ``namespace`` can retrieve, contained in ``tokens``.

        Every candidate is verified token-exact, then narrowed to the
        occupants ``namespace`` stores. Content held only by other
        namespaces yields nothing.

        Args:
            tokens: The query token ids (any dtype castable to
                ``uint64``).
            namespace: The requester's retrieval namespace.

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
                for chunk_hash, occupant in entry.occupants.items():
                    # Skip occupants this namespace cannot retrieve, and
                    # chunks already emitted elsewhere in the query.
                    if chunk_hash in seen or namespace not in occupant.namespaces:
                        continue
                    seen.add(chunk_hash)
                    matches.append(
                        BlendMatch(
                            chunk_hash=chunk_hash,
                            old_st=occupant.token_offset,
                            cur_st=cur_st,
                        )
                    )
                    break  # occupants are content-identical; one suffices
        return matches

    def stats(self) -> BlendIndexStats:
        """Return a point-in-time summary of index contents.

        Returns:
            Distinct contents, chunks, namespace claims, distinct
            namespaces, and the table size.
        """
        with self._lock:
            num_chunks = 0
            num_claims = 0
            namespaces: set[BlendNamespace] = set()
            for entry in self._fingerprint_table.values():
                num_chunks += len(entry.occupants)
                for occupant in entry.occupants.values():
                    num_claims += len(occupant.namespaces)
                    namespaces |= occupant.namespaces
            return BlendIndexStats(
                num_contents=len(self._fingerprint_table),
                num_chunks=num_chunks,
                num_claims=num_claims,
                num_namespaces=len(namespaces),
                table_size=int(self._slots.shape[0]),
            )

    # -- Internals -------------------------------------------------------------

    def _drop_entry_if_empty(self, poly: int, entry: _FingerprintEntry) -> None:
        """Retire ``entry`` once no chunk holds its content. Call with the
        lock held."""
        if entry.occupants:
            return
        del self._fingerprint_table[poly]
        # The bit stays until a rebuild: clearing it here could hide a
        # different fingerprint sharing the bucket.
        if self._bits_set > 2 * len(self._fingerprint_table):
            self._rebuild_table()

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
