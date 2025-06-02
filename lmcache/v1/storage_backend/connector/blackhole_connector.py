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
from typing import List, Optional, no_type_check

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj

# reuse
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class BlackholeConnector(RemoteConnector):
    def __init__(self, config: Optional[LMCacheEngineConfig] = None):
        """
        Initialize the Blackhole connector.

        Args:
            config: LMCacheEngineConfig instance for accessing extra_config
        """
        self.always_exists = False
        if config and config.extra_config:
            self.always_exists = config.extra_config.get(
                "remote_black_hole_always_exists", False
            )

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.always_exists

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        pass

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        logger.info("Closed the blackhole connection")
