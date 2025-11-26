# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class LRUCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    LRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing LRUCachePolicy")

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        cache_dict.move_to_end(key)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        # No action needed for LRU on put, as the key is already at the end.
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
