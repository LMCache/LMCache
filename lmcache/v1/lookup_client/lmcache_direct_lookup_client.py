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

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


class LMCacheDirectLookupClient(LookupClientInterface):
    def __init__(self, lmcache_engine: LMCacheEngine):
        assert lmcache_engine is not None
        self.lmcache_engine = lmcache_engine

    def lookup(self, token_ids: torch.Tensor) -> int:
        return self.lmcache_engine.lookup(token_ids, pin=True)

    def close(self):
        # Nothing to close
        pass
