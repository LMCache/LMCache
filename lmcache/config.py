# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


@dataclass
class LMCacheEngineMetadata:
    """
    LMCacheEngineMetadata should be extracted from the northbound
    serving engine configuration
    """

    """name of the LLM model"""
    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ use case of LMCache (vllm, sglang, standalone, etc.) """
    use_case: str = "vllm"
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
    """ extra config from kv_connector (e.g., lmcache_rpc_port) """
    kv_connector_extra_config: Optional[dict] = None

    serving_engine_config: Optional[Any] = None

    @property
    def fmt(self) -> str:
        """
        Deprecated: Use use_case instead.
        """
        return self.use_case

    @property
    def num_ranks(self) -> int:
        """
        Deprecated: Use world_size instead.
        """
        return self.world_size

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

    """
    "needs" methods determine if LMCacheManager should create
    the corresponding component
    """

    def needs_cache_engine(self, lmcache_config: LMCacheEngineConfig) -> bool:
        if self.use_case == "vllm":
            if self.role == "scheduler":
                # Scheduler only needs engine when BYPASSING the lookup client-server
                # (i.e., when making direct lookups instead of ZMQ lookups)
                # bypass cache engine holds a storage manager with
                # remote backend that queries key existence in remote
                # database (e.g. redis or S3). No GPU connector or
                # Local CPU Backend
                # the backend management is done in storage_backends/__init__.py
                if lmcache_config and lmcache_config.enable_scheduler_bypass_lookup:
                    assert self.save_only_first_rank(
                        lmcache_config
                    ) or lmcache_config.get_extra_config_value(
                        "remote_enable_mla_worker_id_as0", self.use_mla
                    ), (
                        "enable_scheduler_bypass_lookup is only supported with "
                        "save_only_first_rank or remote_enable_mla_worker_id_as0"
                    )
                return (
                    lmcache_config.enable_scheduler_bypass_lookup
                    if lmcache_config
                    else False
                )
            elif self.role == "worker":
                # Workers always need engine (they do the actual caching)
                return True
            else:
                # Unknown role, default to requiring engine
                return True
        else:
            # Currently, only vLLM has use cases that do not require an LMCacheEngine
            # for other use cases, LMCacheManager will require the main service of
            # LMCacheEngine to exist
            logger.warning(
                "disabling LMCacheEngine is not supported "
                f"for use case {self.use_case} yet"
            )
            return True

    @property
    def needs_lookup_client(self) -> bool:
        if self.use_case == "vllm":
            return self.role == "scheduler"
        else:
            # the lookup client (and server) are only used for vLLM for now
            logger.warning(
                f"lookup client is not supported for use case {self.use_case} yet"
            )
            return False

    @property
    def needs_lookup_server(self) -> bool:
        if self.use_case == "vllm":
            return self.role == "worker"
        else:
            # the lookup server (and client) are only used for vLLM for now
            logger.warning(
                f"lookup server is not supported for use case {self.use_case} yet"
            )
            return False

    @property
    def needs_offload_server(self) -> bool:
        if self.use_case == "vllm":
            return self.role == "worker"
        else:
            # the offload server is only used for vLLM for now
            logger.warning(
                f"offload server is not supported for use case {self.use_case} yet"
            )
            return False

    @property
    def is_dp_rank0(self) -> bool:
        if self.use_case == "vllm" and self.serving_engine_config is not None:
            return (
                self.serving_engine_config.parallel_config.data_parallel_rank_local == 0
            )
        else:
            # TODO: add the way to query data parallel rank for other use cases
            logger.warning(
                "data parallel rank checking is not supported "
                f"for use case {self.use_case} yet"
            )
            return False

    @property
    def needs_api_server(self) -> bool:
        if self.use_case == "vllm":
            return self.is_dp_rank0
        elif self.use_case == "standalone":
            # Standalone always has an API server for monitoring
            return True
        else:
            # Other use cases don't have API server by default
            logger.warning(
                f"Internal API server is not supported for use case {self.use_case} yet"
            )
            return False

    @property
    def needs_runtime_plugin_launcher(self) -> bool:
        if self.use_case == "vllm":
            return self.is_dp_rank0
        else:
            # TODO: add the way to query data parallel rank for other use cases
            logger.warning(
                "runtime plugin launcher is not supported "
                f"for use case {self.use_case} yet"
            )
            return False

    @property
    def needs_health_monitor(self) -> bool:
        if self.use_case == "vllm":
            return True
        else:
            # Other use cases don't have health monitor by default
            logger.warning(
                f"health monitor is not supported for use case {self.use_case} yet"
            )
            return False

    # -- Platform detection properties --
    @property
    def is_cuda_alike(self) -> bool:
        """Check if the platform is CUDA-alike (CUDA, ROCm, etc.)."""
        if self.use_case == "vllm":
            try:
                # Third Party
                from vllm.platforms import current_platform

                return current_platform.is_cuda_alike()
            except ImportError:
                logger.warning("vllm not available, falling back to torch.cuda check")
                return torch.cuda.is_available()
        elif self.use_case == "sglang":
            # SGLang currently only supports CUDA
            return torch.cuda.is_available()
        else:
            # Default fallback for other use cases
            return torch.cuda.is_available()

    @property
    def is_xpu(self) -> bool:
        """Check if the platform is XPU (Intel GPU)."""
        if self.use_case == "vllm":
            try:
                # Third Party
                from vllm.platforms import current_platform

                return current_platform.is_xpu()
            except ImportError:
                logger.warning("vllm not available, XPU check unavailable")
                return False
        elif self.use_case == "sglang":
            # SGLang doesn't support XPU currently
            return False
        else:
            return False

    @property
    def compute_device(self) -> Any:  # torch.device
        """Get the compute device dynamically based on use_case."""
        if self.use_case == "vllm":
            if self.serving_engine_config is not None:
                # Get device from vLLM config
                parallel_config = self.serving_engine_config.parallel_config

                # Determine device type based on platform
                try:
                    # Third Party
                    from vllm.platforms import current_platform

                    if current_platform.is_cuda_alike():
                        torch_dev = torch.cuda
                        dev_name = "cuda"
                    elif current_platform.is_xpu():
                        torch_dev = torch.xpu
                        dev_name = "xpu"
                    else:
                        raise RuntimeError("Unsupported device platform")

                    num_gpus = torch_dev.device_count()
                    local_rank = parallel_config.rank % num_gpus
                    return torch.device(f"{dev_name}:{local_rank}")
                except Exception as e:
                    logger.warning(
                        "Failed to get device from vLLM config: %s, "
                        "using default cuda:0",
                        e,
                    )
                    return torch.device("cuda:0")
            else:
                logger.warning(
                    "serving_engine_config not available, using default cuda:0"
                )
                return torch.device("cuda:0")
        elif self.use_case == "sglang":
            # For SGLang, construct device from worker_id
            # SGLang uses worker_id as the device index
            return torch.device(f"cuda:{self.worker_id}")
        elif self.use_case == "standalone":
            # Standalone always uses MockGPUConnector, which doesn't need a real device
            # Return cpu as a safe default (not actually used by MockGPUConnector)
            return torch.device("cpu")
        else:
            # Default fallback
            logger.warning(f"Unknown use_case {self.use_case}, using default cuda:0")
            return torch.device("cuda:0")

    @property
    def engine_name(self) -> str:
        """
        Get the ENGINE_NAME for this use case.

        Returns:
            str: The engine instance name for the LMCacheEngineBuilder
        """
        if self.use_case == "vllm":
            # First Party
            from lmcache.integration.vllm.utils import ENGINE_NAME

            return ENGINE_NAME
        elif self.use_case == "sglang":
            # First Party
            from lmcache.integration.sglang.utils import ENGINE_NAME

            return ENGINE_NAME
        elif self.use_case == "standalone":
            # First Party
            from lmcache.integration.standalone.utils import ENGINE_NAME

            return ENGINE_NAME
        else:
            # Default engine name for other use cases
            return f"{self.use_case}-instance"

    @property
    def broadcast_fn(self):
        """
        Get the tensor broadcast function for distributed communication.

        Returns:
            Callable: Function to broadcast tensors across ranks
        """
        if self.use_case == "vllm":
            if self.role == "scheduler":
                # No-op broadcast for scheduler (no actual tensor parallel)
                return lambda tensor, src: tensor
            else:
                # Real tensor parallel group for workers
                # Third Party
                from vllm.distributed.parallel_state import get_tp_group

                tpg = get_tp_group()
                return tpg.broadcast
        else:
            # Default no-op for other use cases (sglang, standalone)
            # First Party
            from lmcache.utils import mock_up_broadcast_fn

            return mock_up_broadcast_fn

    @property
    def broadcast_object_fn(self):
        """
        Get the object broadcast function for distributed communication.

        Returns:
            Callable: Function to broadcast Python objects across ranks
        """
        if self.use_case == "vllm":
            if self.role == "scheduler":
                # No-op broadcast for scheduler
                return lambda obj, src: obj
            else:
                # Real tensor parallel group for workers
                # Third Party
                from vllm.distributed.parallel_state import get_tp_group

                tpg = get_tp_group()
                return tpg.broadcast_object
        else:
            # Default no-op for other use cases
            # First Party
            from lmcache.utils import mock_up_broadcast_object_fn

            return mock_up_broadcast_object_fn

    # -- Serving engine specific property accessors --
    def save_only_first_rank(self, lmcache_config) -> bool:
        """
        Check if save_only_first_rank is enabled.

        Args:
            lmcache_config: LMCacheEngineConfig instance

        Returns:
            bool: True if save_only_first_rank is enabled (only works with MLA)
        """
        # save_only_first_rank only works when use mla
        return (
            lmcache_config.get_extra_config_value("save_only_first_rank", self.use_mla)
            and self.use_mla
        )

    @property
    def tensor_model_parallel_rank(self) -> int:
        """
        The LMCacheManager uses this for:
        - creating offload server
        """
        if self.use_case == "vllm":
            # Third Party
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        else:
            # TODO: add branching logic for other use cases
            logger.warning(
                "returning default 0 for tensor model parallel rank "
                f"check for use case {self.use_case}"
            )
            return 0

    @property
    def tensor_parallel_size(self) -> int:
        """
        The LMCacheManager uses this for:
        - creating RuntimePluginLauncher
        """
        if self.use_case == "vllm" and self.serving_engine_config is not None:
            return self.serving_engine_config.parallel_config.tensor_parallel_size
        else:
            # TODO: add branching logic for other use cases
            logger.warning(
                "returning default 1 for tensor parallel size "
                f"check for use case {self.use_case}"
            )
            return 1

    @property
    def get_kv_connector_extra_config(self):
        kv_connector_extra_config = None
        if self.use_case == "vllm" and self.serving_engine_config is not None:
            if hasattr(self.serving_engine_config, "kv_transfer_config"):
                kv_transfer_config = self.serving_engine_config.kv_transfer_config
                if kv_transfer_config is not None:
                    _ = getattr(kv_transfer_config, "engine_id", None)
                    kv_connector_extra_config = getattr(
                        kv_transfer_config, "kv_connector_extra_config", None
                    )
        else:
            logger.warning(
                "default None for kv connector extra config "
                f"for use case {self.use_case}"
            )
        return kv_connector_extra_config

    @property
    def _calculate_draft_layers(self) -> int:
        num_draft_layers = 0
        if self.use_case == "vllm" and self.serving_engine_config is not None:
            vllm_config = self.serving_engine_config
            model_config = vllm_config.model_config

            if vllm_config.speculative_config is not None:
                logger.info(
                    "vllm_config.speculative_config: %s", vllm_config.speculative_config
                )
                if vllm_config.speculative_config.method == "deepseek_mtp":
                    num_draft_layers = getattr(
                        model_config.hf_config, "num_nextn_predict_layers", 0
                    )
                elif vllm_config.speculative_config.use_eagle():
                    try:
                        draft_model_config = (
                            vllm_config.speculative_config.draft_model_config
                        )
                        num_draft_layers = draft_model_config.get_num_layers(
                            vllm_config.parallel_config
                        )
                        logger.info(
                            "EAGLE detected %d extra layer(s)", num_draft_layers
                        )
                    except Exception:
                        logger.info(
                            "EAGLE detected, but failed to get the "
                            "number of extra layers, falling back to 1"
                        )
                        num_draft_layers = 1
        else:
            # TODO: add branching logic for other use cases
            logger.warning(
                "returning default 0 for draft layers check "
                f"for use case {self.use_case}"
            )
        return num_draft_layers


@dataclass
class LMCacheMemPoolMetadata:
    """Subset of `LMCacheEngineMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


blend_default_separator = "[BLEND_SEP]"
