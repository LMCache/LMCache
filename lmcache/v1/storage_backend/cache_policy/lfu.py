# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


class LFUCachePolicy(BaseCachePolicy):
    """
    LFU cache policy.
    """
    
    # NOTE(Jiayi): We use `ordered dict` + `bucket` to implement LFU.
    # NOTE(Jiayi): We use FIFO for entries with the same frequency.
    def __init__(self):
        self.freq_to_keys = {}

        # TODO(Jiayi): We can optimize this a bit by using `key_to_val_freq`
        self.key_to_freq = {}

        self.min_freq = 0
        self.max_freq = 0

        logger.info("Initializing LFUCachePolicy")
    
    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(
        self, 
        key: CacheEngineKey, 
        cache_dict: dict[CacheEngineKey, Any],
    ) -> None:
        curr_freq = self.key_to_freq[key]
        self.freq_to_keys[curr_freq].pop(key)

        if curr_freq == self.min_freq:
            self.min_freq += 1
        
        if curr_freq == self.max_freq:
            self.max_freq += 1

        curr_freq += 1
        self.key_to_freq[key] = curr_freq

        if curr_freq not in self.freq_to_keys:
            self.freq_to_keys[curr_freq] = OrderedDict(key=None)
        else:
            self.freq_to_keys[curr_freq][key] = None
        

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys mignt be smaller than num_candidates.
    def get_evict_candidates(
        self, 
        cache_dict: dict[CacheEngineKey, Any], 
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        
        evict_keys = []
        curr_min_freq = self.min_freq

        # Previous `min_freq` whose bucket still has keys.
        prev_min_freq = 0


        # Whether an unfull bucket has been evicted.
        # In this case we don't have to skip pinned keys.
        evict_unfull = False

        while True:
            if curr_min_freq not in self.freq_to_keys:
                curr_min_freq += 1
                continue
            
            if not evict_unfull:
                self.min_freq = curr_min_freq

            fifo_keys = self.freq_to_keys[curr_min_freq]
            
            evict_keys_in_bucket = []
            evict_unfull_this_bucket = False
            for key in fifo_keys:
                if cache_dict[key].is_pinned:
                    continue
                evict_keys_in_bucket.append(key)
                self.key_to_freq.pop(key)
                if len(evict_keys) + len(evict_keys_in_bucket) == num_candidates:
                    break
            
            if len(evict_keys_in_bucket) < len(fifo_keys):
                evict_unfull = True
                evict_unfull_this_bucket = True
            
            for key in evict_keys_in_bucket:
                fifo_keys.pop(key)
                evict_keys.append(key)

            if not fifo_keys:
                self.freq_to_keys.pop(curr_min_freq)
            
            if curr_min_freq == self.max_freq:
                if not evict_unfull_this_bucket:
                    self.max_freq = prev_min_freq
                break
            
            if evict_unfull_this_bucket:
                prev_min_freq = curr_min_freq

            curr_min_freq += 1

        return evict_keys
