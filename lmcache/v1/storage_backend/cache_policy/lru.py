# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Dict
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


class LRUCachePolicy(BaseCachePolicy[OrderedDict[CacheEngineKey, Any]]):
    """
    LRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing LRUCachePolicy")
        self.chunk_hash_to_init_timestamp: Dict[Any, Any] = {}
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.max_num_chunk_hash = 12500000

    def init_mutable_mapping(self) -> OrderedDict[CacheEngineKey, Any]:
        return OrderedDict()

    def update_chunk_hash_dict(self, key: CacheEngineKey) -> None:
        if key.chunk_hash not in self.chunk_hash_to_init_timestamp:
            if len(self.chunk_hash_to_init_timestamp) >= self.max_num_chunk_hash:
                # Clear the dictionary to avoid memory leak
                self.chunk_hash_to_init_timestamp = {}
            self.chunk_hash_to_init_timestamp[key.chunk_hash] = time.time()
        if key.chunk_hash in self.chunk_hash_to_init_timestamp:
            time_interval = (
                time.time() - self.chunk_hash_to_init_timestamp[key.chunk_hash]
            )
            self.stats_monitor.on_chunk_reuse(time_interval)

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: OrderedDict[CacheEngineKey, Any],
    ) -> None:
        self.update_chunk_hash_dict(key)
        cache_dict.move_to_end(key)

    def update_on_put(
        self,
        key: CacheEngineKey,
    ) -> None:
        self.update_chunk_hash_dict(key)
        pass

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        pass

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys mignt be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys
