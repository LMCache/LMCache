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

# First Party
from lmcache.logging import init_logger
from lmcache.v1.server.storage_backend.abstract_backend import LMSBackendInterface
from lmcache.v1.server.storage_backend.local_backend import LMSLocalBackend

logger = init_logger(__name__)


def CreateStorageBackend(
    device: str, max_cache_size: float = 10.0
) -> LMSBackendInterface:
    match device:
        case "cpu":
            # cpu only
            logger.info(
                f"Initializing cpu-only cache server with "
                f"max_cache_size={max_cache_size}GB"
            )
            return LMSLocalBackend(max_cache_size=max_cache_size)
        case _:
            raise ValueError(f"Unsupported device: {device}")
        # TODO(Jiayi): please implement hierarchical remote storage
        # case _:
        #    logger.info("Initializing disk-only cache server")
        #    return LMSLocalDiskBackend(path=device, max_cache_size=max_cache_size)
