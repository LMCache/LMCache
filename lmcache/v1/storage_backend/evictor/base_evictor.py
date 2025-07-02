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
from enum import Enum
from typing import List, Tuple
import abc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class PutStatus(Enum):
    LEGAL = 1
    ILLEGAL = 2


class BaseEvictor(metaclass=abc.ABCMeta):
    """
    Interface for cache evictor
    """

    @abc.abstractmethod
    def update_on_hit(self, key: CacheEngineKey, cache_dict: OrderedDict) -> None:
        """
        Update cache_dict when a cache is used is used

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_on_put(
        self, cache_dict: OrderedDict, cache_size: int
    ) -> Tuple[List[CacheEngineKey], PutStatus]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected

        Return:
            return a list of keys to be evicted and a PutStatus
            to indicate whether the put is allowed
        """
        raise NotImplementedError
