# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class LMCacheEngineMetadata:
    """
    Metadata extracted from the northbound serving engine
    """

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
    """ engine_id for RPC path (used by lookup client/server) """
    engine_id: Optional[str] = None
    """ total number of ranks (tensor_parallel_size * pipeline_parallel_size) """
    num_ranks: int = 1
    """ extra config from kv_connector (e.g., lmcache_rpc_port) """
    kv_connector_extra_config: Optional[dict] = None

    """ device for tensor operations (e.g., cuda:0, xpu:0) """
    device: Optional[torch.device] = None
    """ torch device module for device operations (torch.cuda or torch.xpu) """
    torch_device_module: Optional[Any] = None
    """ device platform name (e.g., 'cuda', 'xpu') """
    device_name: Optional[str] = None
    """ block size for paged attention in serving engine """
    block_size: Optional[int] = None
    """ number of layers in the model (extracted from kv_shape[0]) """
    num_layers: Optional[int] = None
    """ number of KV heads per GPU (extracted from kv_shape[3]) """
    num_kv_heads: Optional[int] = None
    """ head size dimension (extracted from kv_shape[4]) """
    head_size: Optional[int] = None
    """ tensor parallel size """
    tensor_parallel_size: Optional[int] = None
    """ pipeline parallel size """
    pipeline_parallel_size: int = 1  # Default to 1 (no pipeline parallelism)
    """ data parallel local rank (for multi-instance serving) """
    data_parallel_rank_local: int = 0  # Default to 0 (first DP rank)
    """ KV transfer role (e.g., 'kv_producer', 'kv_consumer', None) """
    kv_role: Optional[str] = None
    """ broadcast function for tensors (from serving engine's distributed backend) """
    broadcast_fn: Any = field(default_factory=lambda: lambda tensor, src: tensor)
    """ broadcast function for objects (from serving engine's distributed backend) """
    broadcast_object_fn: Any = field(default_factory=lambda: lambda obj, src: obj)

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank

    def is_cuda_alike(self) -> bool:
        """Check if device is CUDA or CUDA-like"""
        if self.device_name is None:
            return True  # Default to CUDA for backward compatibility
        return self.device_name == "cuda" or self.device_name.startswith("cuda")

    def is_xpu(self) -> bool:
        """Check if device is XPU"""
        return self.device_name == "xpu"

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
