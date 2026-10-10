# SPDX-License-Identifier: Apache-2.0
"""Blend fingerprint matcher: token-level probe over chunk poly-hashes."""

# Standard
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    update_table_id_numba,
)

logger = init_logger(__name__)


class BlendTokenRangeMatcher:
    """Fingerprint matcher: token-level probe (any offset) + full-hash
    collision rejection over a direct-address table of chunk poly-hashes."""

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
        # token_hash -> start position in its registered sequence
        self._token_hash_to_start: dict[bytes, int] = {}
        # compact_chunk_id -> table slot (reverse lookup for eviction)
        self._compact_id_to_slot = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        # token_hash -> compact_chunk_id (for eviction lookup)
        self._token_hash_to_compact_id: dict[bytes, int] = {}
        self._lock = threading.Lock()
        # compact_chunk_id -> full poly hash, for collision reject.
        self._chunk_poly_hash: list[int] = []
        # poly hash -> every live compact_chunk_id holding that content. The
        # same text stored behind different prefixes is one entry per copy;
        # the probe picks the copy nearest the query position (smallest
        # re-RoPE shift).
        self._poly_to_cids: dict[int, list[int]] = {}

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        start_chunk_idx: int = 0,
        position_offset: int = 0,
    ) -> int:
        """Index a stored sequence's non-overlapping chunks. Thread-safe.

        Already-indexed token hashes are skipped; under ``dedup_content``,
        already-indexed poly hashes are skipped too (same text behind
        different prefixes is indexed once).

        Args:
            token_ids: The stored sequence's token IDs.
            token_hashes: Per-chunk content hashes (dedup/eviction key).
            start_chunk_idx: First chunk to index.
            position_offset: Added to each recorded start position.

        Returns:
            Number of chunks newly indexed (0 if all registered, no full
            chunk, or the compact-ID table is full).
        """
        arr = np.array(token_ids, dtype=np.uint64)
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0 or start_chunk_idx >= n:
            return 0

        with self._lock:
            new_idxs: list[int] = []
            batch_poly: set[int] = set()
            for i in range(start_chunk_idx, n):
                if token_hashes[i] in self._token_hash_to_compact_id:
                    continue
                if self._dedup_content:
                    poly_hash = int(chunk_hashes[i])
                    if poly_hash in batch_poly or self._poly_hash_registered(poly_hash):
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
                self._chunk_poly_hash.append(poly_hash)
                self._token_hash_to_start[th] = (
                    position_offset + orig_i * self.chunk_size
                )
                self._compact_id_to_slot[cid] = slot
                self._token_hash_to_compact_id[th] = cid
                self._poly_to_cids.setdefault(poly_hash, []).append(cid)
        return n_new

    def _poly_hash_registered(self, poly_hash: int) -> bool:
        """Whether a live chunk with this poly hash is indexed (bucket-only
        collisions report False). Caller must hold ``self._lock``."""
        cid = int(self._table_id[poly_hash & int(self._mask)])
        if cid < 0:
            return False
        return (
            self._chunk_poly_hash[cid] == poly_hash
            and self._chunk_token_hash[cid] is not None
        )

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Find every registered chunk reused anywhere in a query sequence.

        Vectorized direct-address probe over all token positions, then a
        full poly-hash verify that rejects bucket collisions. Thread-safe.

        When the same content is stored at several positions, each query
        position takes the not-yet-used copy whose stored position is
        nearest, so the re-RoPE shift is as small as possible.

        Returns:
            One result per reused stored chunk (cur_st = query position,
            old_st = stored position); empty if the query is shorter than
            one chunk or nothing matched.
        """
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)

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
                poly_hash = int(rolling[pos])
                if poly_hash != self._chunk_poly_hash[cid]:
                    continue  # bucket-only collision
                picked = self._nearest_copy(poly_hash, pos, seen_cids)
                if picked is None:
                    continue
                cid, th, old_st = picked
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

    def _nearest_copy(
        self, poly_hash: int, pos: int, seen_cids: set[int]
    ) -> tuple[int, bytes, int] | None:
        """The unused live copy of ``poly_hash`` stored nearest ``pos``, as
        ``(cid, token_hash, old_st)``. Caller must hold ``self._lock``."""
        best: tuple[int, bytes, int] | None = None
        for cid in self._poly_to_cids.get(poly_hash, ()):
            if cid in seen_cids:
                continue
            th = self._chunk_token_hash[cid]
            if th is None:
                continue  # evicted
            old_st = self._token_hash_to_start.get(th)
            if old_st is None:
                continue
            if best is None or abs(old_st - pos) < abs(best[2] - pos):
                best = (cid, th, old_st)
        return best

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict the given chunks so later probes cannot match them.
        Thread-safe."""
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get(th)
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
                poly_hash = self._chunk_poly_hash[cid]
                siblings = self._poly_to_cids.get(poly_hash, [])
                if cid in siblings:
                    siblings.remove(cid)
                if not siblings:
                    self._poly_to_cids.pop(poly_hash, None)
                # Only touch the slot while it still points here: it may hold
                # a newer copy of this content, or a colliding content.
                if int(self._table_id[slot]) == cid:
                    self._table_id[slot] = siblings[-1] if siblings else -1
                self._compact_id_to_slot[cid] = -1
                self._chunk_token_hash[cid] = None
                self._chunk_poly_hash[cid] = 0
                self._token_hash_to_start.pop(th, None)
                del self._token_hash_to_compact_id[th]


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Total token coverage, merging overlapping ranges (sliding-window probe
    can return overlaps; naive sum would double-count)."""
    if not results:
        return 0
    intervals = sorted((r.cur_st, r.cur_ed) for r in results)
    coverage = 0
    cur_end = -1
    for st, ed in intervals:
        if st >= cur_end:
            coverage += ed - st
        elif ed > cur_end:
            coverage += ed - cur_end
        cur_end = max(cur_end, ed)
    return coverage
