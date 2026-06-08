# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide CacheBlend fingerprint directory held by the MP coordinator.

Blend mp-servers publish stored token ranges on STORE and query with the request
tokens on LOOKUP; the coordinator does **all hashing and matching**, so the whole
matching algorithm lives here and can evolve without touching or redeploying
servers. Servers send raw tokens plus the storage mapping they alone know
(``object_key`` per chunk, base token position).

The algorithm matches the local matcher (`BlendTokenRangeMatcherV3`): a table of
non-overlapping chunk polynomial hashes, probed by a strided rolling-hash scan of
the request.

- **Match unit = chunk_size** (fleet config). Each stored chunk is one poly hash.
- **Probe stride** (the querying server's inference block size) controls which
  request offsets can seed a match; sent per query, so servers with different
  (per-machine, dynamic) block sizes interoperate.
- **Scope** = ``f"{model_name}@{cache_salt}"``; cross-scope content never matches.
- **TP rank** is resolved at retrieve (``ipc_key_to_object_keys``), not here.

``object_key`` is the chunk's shared-L2 storage key (``th``), which is
prefix-bound and computed by the storing server -- the coordinator cannot derive
it, so it is supplied with each published range.

Thread-safe (single lock, mirroring ``InstanceRegistry``) and ephemeral. A stale
entry (chunk evicted from L2 but not yet removed here) only causes a wasted
prefetch downstream, then recompute -- never wrong KV.

See ``docs/design/v1/mp_coordinator/blend_lookup.md``.
"""

# Standard
from dataclasses import dataclass
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
)

logger = init_logger(__name__)

# Fleet-constant polynomial base for blend fingerprints. The coordinator owns
# the hashing, so this lives here: the same base hashes published chunks
# (``chunk_hash_windows_numba``) and probes requests (``rolling_hash_windows_numba``)
# so the two align. Constant across a coordinator's lifetime.
POLY_BASE = np.uint64(0x9E3779B97F4A7C15)


@dataclass
class StoreRange:
    """One stored token range published by a blend server.

    Attributes:
        model_scope: Reuse-compatibility scope, ``f"{model_name}@{cache_salt}"``.
        tokens: The stored tokens (``token_ids[start:end]``). The coordinator
            chunks these at ``chunk_size`` and hashes each chunk.
        object_keys: Shared-L2 storage key (hex of the ObjectKey chunk hash) per
            chunk, in order; chunk ``i`` maps to ``object_keys[i]``.
        old_st_base: Token position of the range's first token in the stored
            sequence; chunk ``i`` starts at ``old_st_base + i * chunk_size``.
    """

    model_scope: str
    tokens: list[int]
    object_keys: list[str]
    old_st_base: int


@dataclass
class _ChunkLoc:
    """Where a registered chunk lives (anchor index value)."""

    object_key: str
    old_st: int


@dataclass
class GlobalMatch:
    """One matched chunk returned to a querying server.

    Attributes:
        object_key: Shared-L2 storage key of the matched chunk.
        old_st: Token position of the chunk in the stored sequence (re-RoPE).
        cur_st: Token position in the request where the match was found.
    """

    object_key: str
    old_st: int
    cur_st: int


class GlobalBlendMatcher:
    """Thread-safe fleet-wide chunk fingerprint directory.

    Hashes published token ranges into a poly-hash table and matches request
    tokens by a strided rolling-hash probe. All public methods take the internal
    lock, so the directory stays consistent under concurrent
    publish/query/evict.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 1) -> None:
        """Initialize an empty directory.

        Args:
            chunk_size: Tokens per chunk (the LMCache chunk size; fleet config).
                The match unit; must be the same value the storing servers use.
            probe_stride: Positions between match probes. With partial-fill reuse
                any offset is usable, so the default ``1`` (probe every offset)
                gives full recall; raise only to trade recall for coordinator CPU.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is not positive.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if probe_stride < 1:
            raise ValueError(f"probe_stride must be >= 1, got {probe_stride}")
        self._chunk_size = chunk_size
        self._probe_stride = probe_stride
        self._lock = threading.Lock()
        self._index: dict[tuple[str, int], _ChunkLoc] = {}
        # Reverse map for eviction: object_key -> its (scope, poly_hash) keys.
        self._by_key: dict[str, list[tuple[str, int]]] = {}

    def register(self, ranges: list[StoreRange]) -> int:
        """Hash and insert published token ranges (idempotent per chunk).

        The coordinator chunks each range's tokens, hashes each chunk, and maps
        the poly hash to its storage key and position. Re-publishing a
        ``(model_scope, poly_hash)`` already present is a no-op.

        A range whose chunk count (``len(tokens) // chunk_size``) does not equal
        its ``object_keys`` count is **skipped** with an error log: the two are
        1:1 by construction, so a mismatch signals a publisher bug or a
        chunk-size disagreement, and registering the aligned prefix would map
        chunks to the wrong storage keys (a shift) -- worse than dropping it.

        Args:
            ranges: Stored token ranges to register.

        Returns:
            Number of chunk fingerprints newly inserted (excludes idempotent
            skips).
        """
        # Hash every range outside the lock; keep only well-formed ranges.
        prepared: list[tuple[str, np.ndarray, list[str], int]] = []
        for rng in ranges:
            arr = np.array(rng.tokens, dtype=np.uint64)
            polys = chunk_hash_windows_numba(arr, self._chunk_size, POLY_BASE)
            n_chunks = int(polys.shape[0])
            if n_chunks != len(rng.object_keys):
                logger.error(
                    "blend register: %d chunks from %d tokens (chunk_size=%d) "
                    "but %d object_keys for scope %s; skipping range "
                    "(publisher/chunk_size mismatch)",
                    n_chunks,
                    len(rng.tokens),
                    self._chunk_size,
                    len(rng.object_keys),
                    rng.model_scope,
                )
                continue
            prepared.append((rng.model_scope, polys, rng.object_keys, rng.old_st_base))

        inserted = 0
        with self._lock:
            for model_scope, polys, object_keys, old_st_base in prepared:
                for i in range(len(object_keys)):
                    key = (model_scope, int(polys[i]))
                    if key in self._index:
                        continue
                    object_key = object_keys[i]
                    self._index[key] = _ChunkLoc(
                        object_key, old_st_base + i * self._chunk_size
                    )
                    self._by_key.setdefault(object_key, []).append(key)
                    inserted += 1
        return inserted

    def remove(self, object_keys: list[str]) -> int:
        """Evict all fingerprints for the given storage keys.

        Args:
            object_keys: Storage keys of chunks to evict.

        Returns:
            Number of fingerprint entries removed.
        """
        removed = 0
        with self._lock:
            for object_key in object_keys:
                for key in self._by_key.pop(object_key, []):
                    if self._index.pop(key, None) is not None:
                        removed += 1
        return removed

    def match(self, model_scope: str, tokens: list[int]) -> list[GlobalMatch]:
        """Match request tokens against the directory.

        Rolls a chunk-window hash over the request and probes the table every
        ``probe_stride`` positions; a hit is an exact 64-bit poly match (the dict
        key is the full hash). De-duplicates by ``object_key``. Mirrors
        ``BlendTokenRangeMatcherV3.match_sub_sequence``.

        Args:
            model_scope: Scope to match within (``f"{model_name}@{cache_salt}"``).
            tokens: The request tokens.

        Returns:
            Matches in ascending ``cur_st`` order; empty if nothing matched.
        """
        if len(tokens) < self._chunk_size:
            return []
        arr = np.array(tokens, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self._chunk_size, POLY_BASE)
        n_positions = int(rolling.shape[0])
        matches: list[GlobalMatch] = []
        seen: set[str] = set()
        with self._lock:
            for q_pos in range(0, n_positions, self._probe_stride):
                loc = self._index.get((model_scope, int(rolling[q_pos])))
                if loc is None or loc.object_key in seen:
                    continue
                seen.add(loc.object_key)
                matches.append(
                    GlobalMatch(
                        object_key=loc.object_key, old_st=loc.old_st, cur_st=q_pos
                    )
                )
        return matches
