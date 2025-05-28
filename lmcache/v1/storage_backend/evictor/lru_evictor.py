# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import OrderedDict
from typing import List, Tuple, Union

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.evictor.base_evictor import BaseEvictor, PutStatus

logger = init_logger(__name__)


class LRUEvictor(BaseEvictor):
    """
    LRU cache evictor
    """

    def __init__(self, max_cache_size: float = 10.0):
        # The storage size limit (in bytes)
        self.MAX_CACHE_SIZE = int(max_cache_size * 1024**3)

        # TODO(Jiayi): need a way to avoid fragmentation
        # current storage size (in bytes)
        self.current_cache_size = 0.0

    def update_on_hit(
        self, key: Union[CacheEngineKey, str], cache_dict: OrderedDict
    ) -> None:
        cache_dict.move_to_end(key)

    def update_on_put(
        self, cache_dict: OrderedDict, cache_size: int
    ) -> Tuple[List[CacheEngineKey], PutStatus]:
        evict_keys = []
        iter_cache_dict = iter(cache_dict)

        if cache_size > self.MAX_CACHE_SIZE:
            logger.warning("Put failed due to limited cache storage")
            return [], PutStatus.ILLEGAL

        # evict cache until there's enough space
        while cache_size + self.current_cache_size > self.MAX_CACHE_SIZE:
            evict_key = next(iter_cache_dict)
            evict_metadata = cache_dict[evict_key]
            if evict_metadata.is_pinned:
                continue
            evict_cache_size = evict_metadata.size
            self.current_cache_size -= evict_cache_size
            evict_keys.append(evict_key)

        # update cache size
        self.current_cache_size += cache_size
        if len(evict_keys) > 0:
            logger.debug(
                f"Evicting {len(evict_keys)} chunks, "
                f"Current cache size: {self.current_cache_size} bytes, "
                f"Max cache size: {self.MAX_CACHE_SIZE} bytes"
            )
        return evict_keys, PutStatus.LEGAL
