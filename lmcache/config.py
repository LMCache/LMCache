# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class LMCacheEngineMetadata:
    """name of the LLM model"""

    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ whether use MLA"""
    use_mla: bool = False
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0
    served_model_name: Optional[str] = None
    """chunk size"""
    chunk_size: int = 256
    """ Manager for groups of layers with identical KV cache structure """
    kv_layer_groups_manager: KVLayerGroupsManager = field(
        default_factory=KVLayerGroupsManager
    )

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank

    # TODO(chunxiaozheng): some uts do not `build_kv_layer_groups`
    def get_dtypes(self) -> list[torch.dtype]:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return [
                group.dtype for group in self.kv_layer_groups_manager.kv_layer_groups
            ]
        return [self.kv_dtype]

    def get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
        """Get the shapes of the KV cache in LMCache"""
        if num_tokens is None:
            num_tokens = self.chunk_size
        if self.kv_layer_groups_manager.kv_layer_groups:
            shapes = []
            kv_size = 1 if self.use_mla else 2
            for group in self.kv_layer_groups_manager.kv_layer_groups:
                shapes.append(
                    torch.Size(
                        [
                            kv_size,
                            group.num_layers,
                            num_tokens,
                            group.hidden_dim_size,
                        ]
                    )
                )
            return shapes
        else:
            return [
                torch.Size(
                    [
                        self.kv_shape[1],
                        self.kv_shape[0],
                        num_tokens,
                        self.kv_shape[3] * self.kv_shape[4],
                    ]
                )
            ]

    def get_num_groups(self) -> int:
        if self.kv_layer_groups_manager.kv_layer_groups:
            return self.kv_layer_groups_manager.num_groups
        return 1


@dataclass
class LMCacheMemPoolMetadata:
    """Subset of `LMCacheEngineMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


blend_default_separator = "[BLEND_SEP]"
