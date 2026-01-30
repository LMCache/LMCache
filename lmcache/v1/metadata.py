# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class GPUKVFormat:
    """
    Format attributes of the GPU to select
    and initialize GPUConnector
    """

    """ whether use MLA"""
    use_mla: bool = False
    """ Manager for groups of layers with identical KV cache structure """
    kv_layer_groups_manager: KVLayerGroupsManager = field(
        default_factory=KVLayerGroupsManager
    )
    """ K and V stacked into one tensor """
    kv_packed: bool = True
    """ Layers stacked into one tensor """
    layers_packed: bool = False


@dataclass
class LMCacheMetadata:
    """name of the LLM model"""

    model_name: str
    """ global world size when running under a distributed setting 
    (total number of workers)"""
    world_size: int
    """ host world size (workers on active localhost)
    This information can be useful for multi-node
    deployment. Will be the same as world_size 
    in single-node deployments.
    """
    local_world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ host worker id (a gpu bound worker id on active localhost)
    This information can be useful for multi-node deployment. 
    Will be the same as worker_id in single-node deployments.
    """
    local_worker_id: int
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0
    served_model_name: Optional[str] = None
    """chunk size"""
    chunk_size: int = 256
    """ engine_id for RPC path (used by lookup client/server) """
    engine_id: Optional[str] = None
    """ extra config from kv_connector (e.g., lmcache_rpc_port) """
    kv_connector_extra_config: Optional[dict] = None
    """ GPU KV format """
    gpu_kv_format: GPUKVFormat = field(default_factory=GPUKVFormat)

    @property
    def use_mla(self) -> bool:
        return self.gpu_kv_format.use_mla

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank

    # TODO(chunxiaozheng): some uts do not `build_kv_layer_groups`
    def get_dtypes(self) -> list[torch.dtype]:
        kv_layer_groups_manager = self.gpu_kv_format.kv_layer_groups_manager
        if kv_layer_groups_manager.kv_layer_groups:
            return [group.dtype for group in kv_layer_groups_manager.kv_layer_groups]
        return [self.kv_dtype]

    def get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
        """Get the shapes of the KV cache in LMCache"""
        if num_tokens is None:
            num_tokens = self.chunk_size
        kv_layer_groups_manager = self.gpu_kv_format.kv_layer_groups_manager
        if kv_layer_groups_manager.kv_layer_groups:
            shapes = []
            kv_size = 1 if self.use_mla else 2
            for group in kv_layer_groups_manager.kv_layer_groups:
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
        kv_layer_groups_manager = self.gpu_kv_format.kv_layer_groups_manager
        assert kv_layer_groups_manager is not None
        if kv_layer_groups_manager.kv_layer_groups:
            return kv_layer_groups_manager.num_groups
        return 1
