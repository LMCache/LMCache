# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


class S3FIFOCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """
    S3-FIFO cache policy with 3 FIFO queues:
    S (small FIFO), M (main FIFO), G (ghost FIFO), and 2-bit frequency counters.

    - No proactive eviction: actual removal from hot_cache is triggered externally
      (e.g., by MemoryAllocator when allocation fails).
    - Metadata is updated on put/hit; get_evict_candidates() only returns victim keys
      in S->M priority order without modifying metadata.
    """

    def __init__(
        self,
        total_capacity: Optional[int] = None,
        small_ratio: float = 0.1,
    ):
        """
        Args:
            total_capacity: Optional total number of entries for metadata sizing.
                            If None, capacities are only used for G trimming.
            small_ratio: Ratio of S queue size to total_capacity.
                         M queue size = total_capacity - S queue size.
                         G queue size = M queue size.
        """
        self.total_capacity = total_capacity
        self.small_ratio = small_ratio

        # Metadata queues: key -> freq
        self.S: "OrderedDict[CacheEngineKey, int]" = OrderedDict()
        self.M: "OrderedDict[CacheEngineKey, int]" = OrderedDict()
        self.G: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

        # Precompute capacities if total_capacity is given
        if total_capacity:
            self.S_cap = max(1, int(total_capacity * small_ratio))
            self.M_cap = total_capacity - self.S_cap
            self.G_cap = self.M_cap
        else:
            self.S_cap = None
            self.M_cap = None
            self.G_cap = None
        logger.info("Initializing S3FIFOCachePolicy")

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        """Return the actual mapping for storing cache entries."""
        return {}

    def update_on_hit(
        self, key: CacheEngineKey, cache_dict: dict[CacheEngineKey, Any]
    ) -> None:
        """
        Update metadata on cache hit:
        - In S or M: freq = min(freq+1, 3).
        - In G: do nothing (ghost hit is not a real hit).
        """
        if key in self.S:
            # In S: increment freq, max 3
            self.S[key] = min(self.S[key] + 1, 3)
        elif key in self.M:
            # In M: increment freq, max 3
            self.M[key] = min(self.M[key] + 1, 3)
        # Ghost hit: not a real hit in S3-FIFO; handled in update_on_put if re-inserted.

    def update_on_put(self, key: CacheEngineKey) -> None:
        """
        Update metadata on new cache insertion:
        - If key in G: remove from G, insert into head of M.
        - Else: insert into head of S.
        - Enforce S capacity: move oldest from S to M (freq>1) or G (freq<=1).
        - No need to enforce G capacity, evict the overflow keys later.
        """
        # If already in cache, no need to insert
        if key in self.S or key in self.M:
            return

        if not self._is_s3fifo():
            # No capacity set - just insert into S as FIFO
            self.S[key] = 0
        else:
            # Capacity is set - use normal logic
            if key in self.G:
                # Ghost hit -> remove from G, insert to head of M
                self.G.pop(key, None)
                self.M[key] = 0
            else:
                # New key -> insert to head of S
                self.S[key] = 0

            # If S overflow, evict oldest to M or G
            if self.S_cap and len(self.S) > self.S_cap:
                self._evictS()

            # If M overflow, evict from M
            while self.M_cap and len(self.M) > self.M_cap:
                self._evictM()

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        """
        Clean up metadata after actual eviction from hot_cache.
        This is called externally after hot_cache.pop(key).
        """
        self.S.pop(key, None)
        self.M.pop(key, None)
        self.G.pop(key, None)

    def get_evict_candidates(
        self, cache_dict: dict[CacheEngineKey, Any], num_candidates: int = 1
    ) -> List[CacheEngineKey]:
        """
        Return up to num_candidates victim keys without modifying metadata:
        - First check G queue excess part (FIFO)
        - Then check M queue excess part (FIFO)
        - Finally search in G->S->M order for all queues
        """
        evict_keys: List[CacheEngineKey] = []

        # 1. First check G queue excess part (FIFO)
        # that's because this part should have been evicted when update_on_put() and
        # the G queue should not have cached any memory objects
        # according to official S3FIFO
        if self.G_cap:
            g_items = list(self.G.keys())
            g_excess_count = max(0, len(g_items) - self.G_cap)
            for i in range(g_excess_count):
                key = g_items[i]
                if len(evict_keys) >= num_candidates:
                    break
                if cache_dict[key].can_evict:
                    evict_keys.append(key)

        # 2. Then check M queue excess part (FIFO)
        # that's because the memory object evicted by M queue should have been
        # deallocated when update_on_put() according to official S3FIFO
        if len(evict_keys) < num_candidates and self.M_cap:
            m_items = list(self.M.keys())
            m_excess_count = max(0, len(m_items) - self.M_cap)
            for i in range(m_excess_count):
                key = m_items[i]
                if len(evict_keys) >= num_candidates:
                    break
                if cache_dict[key].can_evict:
                    evict_keys.append(key)

        # 3. Finally search in G->S->M order for all queues
        # for the priority of each queue
        if len(evict_keys) < num_candidates:
            for q in [self.G, self.S, self.M]:
                items = list(q.keys())
                for key in items:
                    if len(evict_keys) >= num_candidates:
                        break
                    if key not in evict_keys and cache_dict[key].can_evict:
                        evict_keys.append(key)

        # cache is full when calling get_evict_candidates
        # so we can retrieve cache size indirectly
        # segmented s3 fifo queue size can be confirmed now
        self._transform_to_s3fifo(cache_dict)

        # remove key from metadata S3FIFO queue
        for key in evict_keys:
            self.S.pop(key, None)
            self.M.pop(key, None)
            self.G.pop(key, None)

        return evict_keys

    # ----------------------
    # Internal helper methods
    # ----------------------
    def _evictS(self) -> None:
        """
        Evict from S queue according to S3FIFO algorithm:
        - If t.freq > 1: insert t to M
        - If t.freq <= 1: insert t to G and mark as evicted
        """
        if not self.S:
            return

        # Get oldest entry from S (tail)
        old_key, freq = self.S.popitem(last=False)

        if freq > 1:
            # Insert to M
            self.M[old_key] = freq
            # If M is full, evict from M
            while self.M_cap and len(self.M) > self.M_cap:
                self._evictM()
        else:
            # Insert to G
            self.G[old_key] = None

    def _evictM(self) -> None:
        """
        Evict from M queue with FIFO-Reinsertion logic:
        - If t.freq > 0: reinsert t to head of M with freq-1
        - If t.freq = 0: remove t from M and mark as evicted
        """
        if not self.M:
            return

        # Get oldest entry from M (tail)
        evict_key, evict_freq = self.M.popitem(last=False)
        if evict_freq > 0:
            # Reinsert at head with freq-1
            self.M[evict_key] = evict_freq - 1
        # If freq == 0, the key is evicted and not reinserted

    def _update_dynamic_caps(self, cache_dict: dict[CacheEngineKey, Any]) -> None:
        """Recalculate S/M/G capacities based on current hot_cache size."""
        if self._is_s3fifo():
            return  # fixed mode
        if not cache_dict:
            return
        total_size = max(1, len(cache_dict))
        self.S_cap = max(1, int(total_size * self.small_ratio))
        self.M_cap = max(1, total_size - self.S_cap)
        self.G_cap = self.M_cap

    def _transform_to_s3fifo(self, cache_dict: dict[CacheEngineKey, Any]) -> None:
        if self._is_s3fifo():
            return  # fixed mode
        self._update_dynamic_caps(cache_dict)
        # segment original FIFO queue to S/M queue
        for _ in range(self.M_cap):
            old_key, freq = self.S.popitem(last=False)
            self.M[old_key] = freq

    def _is_s3fifo(self) -> bool:
        """Fixed capacity needs set for S3FIFO, fallback to FIFO otherwise."""
        return self.total_capacity is not None

    # ----------------------
    # Debug/Stats
    # ----------------------
    def stats(self) -> Dict[str, int]:
        """Return the sizes of S, M, G for debugging."""
        return {
            "S": len(self.S),
            "M": len(self.M),
            "G": len(self.G),
            "S_cap": self.S_cap or -1,
            "M_cap": self.M_cap or -1,
            "G_cap": self.G_cap or -1,
        }
