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
from typing import Optional
import abc

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj


class DistributedServerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def handle_get(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Handle get from the peer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def issue_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Perform get from the peer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        """
        Start the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Close the server.
        """
        raise NotImplementedError
