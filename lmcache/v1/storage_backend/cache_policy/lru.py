# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Dict, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class LRUCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    LRU cache policy.

    Optional SSD write gating (length and/or access frequency) is configured
    via ``ssd_gate_min_size_bytes`` and ``ssd_gate_min_access_count``. When
    both are zero, all chunks are admitted to disk (no gating).
    """

    def __init__(
        self,
        *,
        ssd_gate_min_size_bytes: int = 0,
        ssd_gate_min_access_count: int = 0,
    ) -> None:
        logger.info("Initializing LRUCachePolicy")
        self.chunk_hash_to_init_timestamp: Dict[Any, float] = {}
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.max_num_chunk_hash = 12500000
        self._ssd_gate_min_size_bytes = ssd_gate_min_size_bytes
        self._ssd_gate_min_access_count = ssd_gate_min_access_count
        # Per chunk_hash access count for frequency-based SSD gating (disk backend).
        self._disk_chunk_access_count: Dict[Any, int] = {}

    @staticmethod
    def _chunk_hash(key: KeyType) -> Any:
        if isinstance(key, CacheEngineKey):
            return key.chunk_hash
        return key

    def disk_access_get(self, key: KeyType) -> int:
        return self._disk_chunk_access_count.get(self._chunk_hash(key), 0)

    def disk_access_increment(self, key: KeyType) -> None:
        h = self._chunk_hash(key)
        self._disk_chunk_access_count[h] = self._disk_chunk_access_count.get(h, 0) + 1

    def disk_access_reset(self, key: KeyType) -> None:
        self._disk_chunk_access_count[self._chunk_hash(key)] = 0

    def disk_access_pop(self, key: KeyType) -> None:
        self._disk_chunk_access_count.pop(self._chunk_hash(key), None)

    def disk_gate_block_reason(
        self,
        key: KeyType,  # noqa: ARG002
        size_bytes: int,
        access_count: int,
    ) -> Optional[str]:
        if (
            self._ssd_gate_min_size_bytes > 0
            and size_bytes < self._ssd_gate_min_size_bytes
        ):
            return "length"
        if (
            self._ssd_gate_min_access_count > 0
            and access_count < self._ssd_gate_min_access_count
        ):
            return "frequency"
        return None

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def update_chunk_hash_dict(self, key: KeyType) -> None:
        curr_time = time.time()
        # HACK: doing type conversion here
        key_hash: Any = key
        if isinstance(key, CacheEngineKey):
            key_hash = key.chunk_hash

        if init_timestamp := self.chunk_hash_to_init_timestamp.get(key_hash, None):
            time_interval = curr_time - init_timestamp
            self.stats_monitor.on_chunk_reuse(time_interval)
        else:
            if len(self.chunk_hash_to_init_timestamp) >= self.max_num_chunk_hash:
                self.chunk_hash_to_init_timestamp.clear()
            self.chunk_hash_to_init_timestamp[key_hash] = curr_time

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        self.update_chunk_hash_dict(key)
        cache_dict.move_to_end(key)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        self.update_chunk_hash_dict(key)
        pass

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        pass

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys mignt be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys
