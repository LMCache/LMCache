# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Dict, Optional

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)


class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}
    _pending_creations: Dict[str, Dict] = {}

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
        If the model is not yet registered, defers creation until register_model is called.
        """

        if instance_id in cls._blenders:
            logger.info(
                f"Blender for {instance_id} already exists, returning the original one."
            )
            return cls._blenders[instance_id]

        try:
            vllm_model = VLLMModelTracker.get_model(instance_id)
        except ValueError:
            logger.info(
                f"Model for {instance_id} not yet registered, "
                "deferring blender creation."
            )
            cls._pending_creations[instance_id] = {
                "cache_engine": cache_engine,
                "gpu_connector": gpu_connector,
                "config": config,
            }
            return None

        logger.info(f"Creating blender for {instance_id}")
        blender = LMCBlender(
            cache_engine=cache_engine,
            gpu_connector=gpu_connector,
            vllm_model=vllm_model,
            config=config,
        )
        cls._blenders[instance_id] = blender
        return blender

    @classmethod
    def register_model(cls, instance_id: str, vllm_model: nn.Module) -> None:
        """
        Register a vllm model and create any pending blenders.
        """
        VLLMModelTracker.register_model(instance_id, vllm_model)

        if instance_id in cls._pending_creations:
            params = cls._pending_creations.pop(instance_id)
            logger.info(f"Creating deferred blender for {instance_id}")
            blender = LMCBlender(
                cache_engine=params["cache_engine"],
                gpu_connector=params["gpu_connector"],
                vllm_model=vllm_model,
                config=params["config"],
            )
            cls._blenders[instance_id] = blender

    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> Optional[LMCBlender]:
        """
        Get the blender by instance_id.
        """
        return cls._blenders.get(instance_id)
