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
        """Initialize the matcher.

        Args:
            chunk_size (int): Tokens per non-overlapping fingerprint chunk.
            dedup_content (bool): When True, skip registering a chunk whose
                poly hash is already indexed (``--enable-dedup-content``).
        """
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

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        start_chunk_idx: int = 0,
        position_offset: int = 0,
    ) -> int:
        """Index a stored sequence's non-overlapping chunks into the matcher.

        Records each new chunk's poly hash + start position so a later
        match_sub_sequence can find it. Chunks whose token hash is already
        indexed are skipped (dedup), so a re-store of the same content
        registers nothing; under ``dedup_content``, already-indexed poly
        hashes are skipped too, so the same text behind different prefixes is
        indexed once. Thread-safe (holds the matcher lock).

        Args:
            token_ids (list[int]): The stored sequence's token IDs.
            token_hashes (list[bytes]): Per-chunk content hashes (one per
                chunk), used as the dedup/eviction key.
            start_chunk_idx (int): First chunk to index; 1 skips chunk 0 (the
                prefix lookup leg owns it).
            position_offset (int): Added to each recorded start position (for
                indexing a tail-slice of a larger sequence).

        Returns:
            int: The number of chunks newly indexed by this call -- ``0`` when
            every chunk was already registered, the range holds no full
            chunk, or the compact-ID table is full.
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
        return n_new

    def _poly_hash_registered(self, poly_hash: int) -> bool:
        """Whether a live chunk with this poly hash is indexed.

        Same probe + full-hash verify as :meth:`match_sub_sequence`, so a
        bucket-only collision reports False. Caller must hold ``self._lock``.

        Args:
            poly_hash (int): The chunk's poly hash over its raw token IDs.

        Returns:
            bool: True if the slot holds a non-evicted chunk with this hash.
        """
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

        Vectorized direct-address probe over all token positions, then a small
        verify loop over the surviving hits (a full poly-hash check rejects
        bucket collisions; evicted/unknown chunks are skipped). Thread-safe.

        Args:
            token_ids (list[int]): The query sequence's token IDs.

        Returns:
            list[CBMatchResult]: One result per unique reused chunk (cur_st
            = its first query position, old_st = its stored position).
            Empty if the query is shorter than one chunk or nothing matched.
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
                if cid in seen_cids:
                    continue
                if int(rolling[pos]) != self._chunk_poly_hash[cid]:
                    continue  # bucket-only collision
                th = self._chunk_token_hash[cid]
                if th is None:
                    continue  # evicted
                old_st = self._token_hash_to_start.get(th)
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

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict the given chunks from the matcher.

        Clears each chunk's table slot + poly hash so later probes cannot match
        it. Thread-safe.

        Args:
            token_hashes (list[bytes]): Content hashes of the chunks to evict.
        """
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
                self._table_id[slot] = -1
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
