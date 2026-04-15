# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Dict

# Third Party
from torch import nn

# First Party
from lmcache.compute.blend.blender import LMCBlender
from lmcache.compute.models.utils import VLLMModelTracker
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.cache_engine import LMCacheEngine
    from lmcache.config import LMCacheEngineConfig
    from lmcache.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)


class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: "LMCacheEngine",
        gpu_connector: "GPUConnectorInterface",
        config: "LMCacheEngineConfig",
    ):
        """
        Get or create a blender for the given instance_id.
        """

        if instance_id not in cls._blenders:
            logger.info(f"Creating blender for {instance_id}")
            vllm_model = VLLMModelTracker.get_model(instance_id)
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
                config=config,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(
                f"Blender for {instance_id} already exists, returning the original one."
            )
        return cls._blenders[instance_id]

    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the blender by instance_id.
        """
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]
