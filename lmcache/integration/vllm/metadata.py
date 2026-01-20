# SPDX-License-Identifier: Apache-2.0
"""
vLLM Metadata Builder for LMCache.

This module contains the logic to construct LMCacheEngineMetadata
from vLLM configuration, isolating vLLM-specific dependencies.
"""

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import mla_enabled
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class VLLMMetadataBuilder:
    """Builder for creating LMCacheEngineMetadata from vLLM configuration."""

    @staticmethod
    def build(
        vllm_config: "VllmConfig",
        lmcache_config: LMCacheEngineConfig,
        role: str,
    ) -> LMCacheEngineMetadata:
        """
        Build LMCacheEngineMetadata from vLLM configuration.

        Args:
            vllm_config: vLLM configuration object
            lmcache_config: LMCache engine configuration
            role: The role string ("scheduler" or "worker")

        Returns:
            LMCacheEngineMetadata instance with vLLM-specific configuration
        """
        # Third Party
        from vllm.platforms import current_platform

        try:
            # Third Party
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            # Third Party
            from vllm.utils import get_kv_cache_torch_dtype

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config

        # Get KV dtype
        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        # Check if MLA is enabled
        use_mla = mla_enabled(model_config)

        # Validate MLA configuration
        VLLMMetadataBuilder._validate_mla_config(use_mla, lmcache_config)

        # Calculate number of layers
        num_layer = model_config.get_num_layers(parallel_config)
        num_draft_layers = VLLMMetadataBuilder._calculate_draft_layers(vllm_config)
        num_layer += num_draft_layers

        # Get KV head configuration
        chunk_size = lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()

        # Construct KV shape
        kv_shape = (
            num_layer,
            1 if use_mla else 2,
            chunk_size,
            num_kv_head,
            head_size,
        )

        logger.info(
            "Building vLLM metadata: num_layer=%d, chunk_size=%d, "
            "num_kv_head=%d, head_size=%d, hidden_dim=%d, use_mla=%s, "
            "kv_shape=%s, num_draft_layers=%d",
            num_layer,
            chunk_size,
            num_kv_head,
            head_size,
            num_kv_head * head_size,
            use_mla,
            kv_shape,
            num_draft_layers,
        )

        # Determine device
        _ = VLLMMetadataBuilder._get_device(vllm_config, current_platform)

        # Extract engine_id and kv_connector_extra_config
        engine_id = None
        kv_connector_extra_config = None
        if hasattr(vllm_config, "kv_transfer_config"):
            kv_transfer_config = vllm_config.kv_transfer_config
            if kv_transfer_config is not None:
                engine_id = getattr(kv_transfer_config, "engine_id", None)
                kv_connector_extra_config = getattr(
                    kv_transfer_config, "kv_connector_extra_config", None
                )

        # Create metadata
        metadata = LMCacheEngineMetadata(
            use_case="vllm",
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            worker_id=parallel_config.rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=role,
            served_model_name=model_config.served_model_name,
            chunk_size=chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
        )

        # Store reference to serving engine config (metadata will wrap any extraction)
        metadata.serving_engine_config = vllm_config

        return metadata

    @staticmethod
    def _calculate_draft_layers(vllm_config: "VllmConfig") -> int:
        """Calculate the number of draft layers for speculative decoding."""
        num_draft_layers = 0
        model_config = vllm_config.model_config

        if vllm_config.speculative_config is not None:
            logger.info(
                "vllm_config.speculative_config: %s",
                vllm_config.speculative_config,
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
                    logger.info("EAGLE detected %d extra layer(s)", num_draft_layers)
                except Exception:
                    logger.info(
                        "EAGLE detected, but failed to get the number of extra layers, "
                        "falling back to 1"
                    )
                    num_draft_layers = 1

        return num_draft_layers

    @staticmethod
    def _get_device(vllm_config: "VllmConfig", current_platform) -> torch.device:
        """Get the compute device based on platform."""
        if current_platform.is_cuda_alike():
            logger.info("CUDA device is available. Using CUDA for LMCache engine.")
            torch_dev = torch.cuda
            dev_name = "cuda"
        elif current_platform.is_xpu():
            logger.info("XPU device is available. Using XPU for LMCache engine.")
            torch_dev = torch.xpu
            dev_name = "xpu"
        else:
            raise RuntimeError("Unsupported device platform for LMCache engine.")

        num_gpus = torch_dev.device_count()
        local_rank = vllm_config.parallel_config.rank % num_gpus
        torch_dev.set_device(local_rank)
        device = torch.device(f"{dev_name}:{local_rank}")

        return device

    @staticmethod
    def _validate_mla_config(
        use_mla: bool, lmcache_config: LMCacheEngineConfig
    ) -> None:
        """Validate MLA-related configuration."""
        if use_mla and (
            lmcache_config.remote_serde != "naive"
            and lmcache_config.remote_serde is not None
        ):
            raise ValueError("MLA only works with naive serde mode.")

        if use_mla and lmcache_config.use_layerwise and lmcache_config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )
