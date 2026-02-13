# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LMCacheMemPoolMetadata:
    """Subset of `LMCacheMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


blend_default_separator = "[BLEND_SEP]"
