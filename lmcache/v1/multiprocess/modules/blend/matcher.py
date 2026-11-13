# SPDX-License-Identifier: Apache-2.0
"""Blend fingerprint matcher: token-level probe over chunk poly-hashes."""

# Standard
from collections.abc import Hashable
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.custom_types import CBMatchResult
from lmcache.v1.common.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    update_table_id_numba,
)

logger = init_logger(__name__)


def _namespace_hash(namespace: Hashable | None) -> np.uint64:
    """Mix process-local namespace identity into the existing bounded index."""
    return np.uint64(0 if namespace is None else hash(namespace) & ((1 << 64) - 1))


class BlendTokenRangeMatcher:
    """Fingerprint matcher: token-level probe (any offset) + full-hash
    collision rejection over a direct-address table of chunk poly-hashes.

    Namespaces share the fixed-size index; collisions can replace candidates,
    but namespace equality is checked before returning any match."""

    _TABLE_BITS: int = 20  # 2^20 ~ 1 M entries
    _TABLE_SIZE: int = 1 << _TABLE_BITS
    _BASE: np.uint64 = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci-hashing const

    def __init__(self, chunk_size: int = 256, dedup_content: bool = False):
        """Initialize with the chunk size (tokens per fingerprint chunk) and
        whether to skip registering already-indexed poly hashes."""
        self.chunk_size = chunk_size
        self._dedup_content = dedup_content
        # poly_chunk_hash -> compact_chunk_id; -1 = empty
        self._table_id = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        self._mask = np.uint64(self._TABLE_SIZE - 1)
        # compact_chunk_id -> caller token_hash (full bytes); None once evicted
        self._chunk_token_hash: list[bytes | None] = []
        self._chunk_namespace: list[Hashable | None] = []
        # (namespace, token_hash) -> start position in its registered sequence
        self._token_hash_to_start: dict[tuple[Hashable | None, bytes], int] = {}
        # compact_chunk_id -> table slot (reverse lookup for eviction)
        self._compact_id_to_slot = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        # (namespace, token_hash) -> compact_chunk_id (for eviction lookup)
        self._token_hash_to_compact_id: dict[tuple[Hashable | None, bytes], int] = {}
        self._lock = threading.Lock()
        # compact_chunk_id -> full poly hash, for collision reject.
        self._chunk_poly_hash: list[int] = []

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        position_offset: int = 0,
        *,
        namespace: Hashable | None = None,
    ) -> int:
        """Index a stored sequence's non-overlapping chunks. Thread-safe.

        Already-indexed token hashes are skipped; under ``dedup_content``,
        already-indexed poly hashes are skipped too (same text behind
        different prefixes is indexed once).

        Args:
            token_ids: The stored sequence's token IDs.
            token_hashes: Per-chunk content hashes (dedup/eviction key).
            position_offset: Added to each recorded start position.
            namespace: Immutable, hashable matching domain. Omitted calls share
                the legacy domain. Deduplication is scoped to this domain.

        Returns:
            Number of chunks newly indexed (0 if all registered, no full
            chunk, or the compact-ID table is full).
        """
        arr = np.array(token_ids, dtype=np.uint64)
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        chunk_hashes ^= _namespace_hash(namespace)
        n = int(chunk_hashes.shape[0])
        if n == 0:
            return 0

        with self._lock:
            new_idxs: list[int] = []
            batch_poly: set[int] = set()
            for i in range(n):
                if (namespace, token_hashes[i]) in self._token_hash_to_compact_id:
                    continue
                if self._dedup_content:
                    poly_hash = int(chunk_hashes[i])
                    if poly_hash in batch_poly or self._poly_hash_registered(
                        poly_hash, namespace
                    ):
                        continue
                    batch_poly.add(poly_hash)
                new_idxs.append(i)
            if not new_idxs:
                return 0
            n_new = len(new_idxs)
            new_chunk_hashes = chunk_hashes[new_idxs]

            base_id = len(self._chunk_token_hash)
            if base_id + n_new > self._TABLE_SIZE:
                logger.error(
                    "BlendTokenRangeMatcher compact-ID overflow: %d chunks "
                    "registered, cannot add %d more (limit %d). Skipping.",
                    base_id,
                    n_new,
                    self._TABLE_SIZE,
                )
                return 0
            if base_id + n_new > int(self._TABLE_SIZE * 0.8):
                logger.warning(
                    "BlendTokenRangeMatcher nearing capacity: %d/%d "
                    "compact IDs used. Hash collision rate is rising; "
                    "hit rate will degrade.",
                    base_id + n_new,
                    self._TABLE_SIZE,
                )
            compact_ids = np.arange(base_id, base_id + n_new, dtype=np.int64)

            update_table_id_numba(new_chunk_hashes, self._table_id, compact_ids)

            for k, orig_i in enumerate(new_idxs):
                th = token_hashes[orig_i]
                cid = int(compact_ids[k])
                poly_hash = int(new_chunk_hashes[k])
                slot = poly_hash & int(self._mask)
                self._chunk_token_hash.append(th)
                self._chunk_namespace.append(namespace)
                self._chunk_poly_hash.append(poly_hash)
                self._token_hash_to_start[namespace, th] = (
                    position_offset + orig_i * self.chunk_size
                )
                self._compact_id_to_slot[cid] = slot
                self._token_hash_to_compact_id[namespace, th] = cid
        return n_new

    def _poly_hash_registered(self, poly_hash: int, namespace: Hashable | None) -> bool:
        """Whether a live chunk with this poly hash is indexed (bucket-only
        collisions report False). Caller must hold ``self._lock``."""
        cid = int(self._table_id[poly_hash & int(self._mask)])
        if cid < 0:
            return False
        return (
            self._chunk_poly_hash[cid] == poly_hash
            and self._chunk_namespace[cid] == namespace
            and self._chunk_token_hash[cid] is not None
        )

    def match_sub_sequence(
        self,
        token_ids: list[int],
        *,
        namespace: Hashable | None = None,
    ) -> list[CBMatchResult]:
        """Find every registered chunk reused anywhere in a query sequence.

        Vectorized direct-address probe over all token positions, then a
        full poly-hash verify that rejects bucket collisions. Thread-safe.

        Args:
            token_ids: Query token sequence.
            namespace: The matching domain used when registering its chunks.

        Returns:
            One result per unique reused chunk (cur_st = first query
            position, old_st = stored position); empty if the query is
            shorter than one chunk or nothing matched.
        """
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)
        rolling ^= _namespace_hash(namespace)

        with self._lock:
            if not self._chunk_token_hash:
                return []

            # Vectorized direct-address probe over all positions. The table is
            # sparse (TABLE_SIZE >> registered chunks), so only true matches and
            # a few bucket collisions reach the Python verify loop below.
            cids_at_pos = self._table_id[rolling & self._mask]
            hit_positions = np.nonzero(cids_at_pos >= 0)[0]

            seen_cids: set[int] = set()
            results: list[CBMatchResult] = []
            for pos in hit_positions:
                pos = int(pos)
                cid = int(cids_at_pos[pos])
                if cid in seen_cids or self._chunk_namespace[cid] != namespace:
                    continue
                if int(rolling[pos]) != self._chunk_poly_hash[cid]:
                    continue  # bucket-only collision
                th = self._chunk_token_hash[cid]
                if th is None:
                    continue  # evicted
                old_st = self._token_hash_to_start.get((namespace, th))
                if old_st is None:
                    continue
                seen_cids.add(cid)
                results.append(
                    CBMatchResult(
                        old_st=old_st,
                        old_ed=old_st + self.chunk_size,
                        cur_st=pos,
                        cur_ed=pos + self.chunk_size,
                        hash=th,
                    )
                )
            logger.info(
                "[match_probe] n_tok=%d table_hits=%d matches=%d",
                len(token_ids),
                len(hit_positions),
                len(results),
            )
            return results

    def remove_chunks(
        self, token_hashes: list[bytes], *, namespace: Hashable | None = None
    ) -> None:
        """Evict the given chunks from their registration domain. Thread-safe.

        Args:
            token_hashes: Hashes to remove from the index.
            namespace: The domain used when registering the chunks.
        """
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get((namespace, th))
                if cid is None:
                    continue
                slot = int(self._compact_id_to_slot[cid])
                if slot < 0:
                    logger.warning(
                        "compact_id %d has no valid table slot; "
                        "entry may have been evicted twice",
                        cid,
                    )
                    continue
                self._table_id[slot] = -1
                self._compact_id_to_slot[cid] = -1
                self._chunk_token_hash[cid] = None
                self._chunk_namespace[cid] = None
                self._chunk_poly_hash[cid] = 0
                self._token_hash_to_start.pop((namespace, th), None)
                del self._token_hash_to_compact_id[namespace, th]


def merge_intervals(results: list[CBMatchResult]) -> list[tuple[int, int]]:
    """Union of the results' ``[cur_st, cur_ed)`` ranges as sorted, disjoint
    intervals; overlapping and abutting ranges merge."""
    merged: list[tuple[int, int]] = []
    for st, ed in sorted((r.cur_st, r.cur_ed) for r in results):
        if merged and st <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ed))
        else:
            merged.append((st, ed))
    return merged


def covered_chunks(results: list[CBMatchResult], chunk_size: int) -> set[int]:
    """Indices of the ``chunk_size``-aligned chunks whose tokens all lie inside
    the union of the results' ranges."""
    return {
        c // chunk_size
        for st, ed in merge_intervals(results)
        for c in range(
            -(-st // chunk_size) * chunk_size, ed - chunk_size + 1, chunk_size
        )
    }


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Total token coverage, merging overlapping ranges (sliding-window probe
    can return overlaps; naive sum would double-count)."""
    return sum(ed - st for st, ed in merge_intervals(results))
