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
from lmcache.config import (
    LMCacheEngineConfig,
    LMCacheEngineMetadata,
    LMCacheMemPoolMetadata,
)
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.hybrid_backend import (  # , LMCPipelinedHybridBackend
    LMCHybridBackend,
)
from lmcache.storage_backend.local_backend import (
    LMCLocalBackend,
    LMCLocalDiskBackend,
)
from lmcache.storage_backend.remote_backend import LMCRemoteBackend

logger = init_logger(__name__)


def CreateStorageBackend(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    dst_device: str = "cuda",
) -> LMCBackendInterface:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    mpool_metadata = LMCacheMemPoolMetadata(
        metadata.kv_shape, metadata.kv_dtype, config.max_local_cache_size
    )
    match config:
        case LMCacheEngineConfig(_, local_device=None, remote_url=str(p)) if (
            p is not None
        ):
            # remote only
            logger.info("Initializing remote-only backend")
            return LMCRemoteBackend(config, metadata, dst_device)

        case LMCacheEngineConfig(_, local_device=str(p), remote_url=None) if (
            p is not None
        ):
            # local only
            match config.local_device:
                case "cpu" | "cuda":
                    logger.info(
                        f"Initializing local-only ({config.local_device}) backend"
                    )

                    return LMCLocalBackend(config, mpool_metadata, dst_device)
                case _:
                    logger.info(
                        f"Initializing local-only (disk) backend at"
                        f" {config.local_device}"
                    )
                    return LMCLocalDiskBackend(config, mpool_metadata, dst_device)

        case LMCacheEngineConfig(_, local_device=str(p), remote_url=str(q)) if (
            p is not None and q is not None
        ):
            logger.info("Initializing hybrid backend")
            return LMCHybridBackend(config, metadata, mpool_metadata, dst_device)

        case _:
            raise ValueError(f"Invalid configuration: {config}")


# __all__ = [
#    "LMCBackendInterface",
#    "LMCLocalBackend",
#    "LMCRemoteBackend",
# ]
