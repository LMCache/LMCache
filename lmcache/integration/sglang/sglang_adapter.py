# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional

# Third Party
from sglang.srt.configs.model_config import ModelConfig
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import (
    SGLangGPUConnector,
)

logger = init_logger(__name__)


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.enable_nixl:
        return False
    else:
        return True


def init_lmcache_engine(
    model_config: ModelConfig,
    tp_size: int,
    rank: int,
    world_size: int,
    kv_dtype: torch.dtype,
) -> LMCacheEngine:
    """
    TODO: ADD COMMENTS
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    config = lmcache_get_config()
    assert isinstance(config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    # construct kv shape (for mem pool)
    num_layer = model_config.num_hidden_layers
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(tp_size)
    head_dim = model_config.head_dim

    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_dim)

    # Change current device.
    torch.cuda.device(rank)
    device = torch.device(f"cuda:{rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model_path,
        world_size,
        rank,
        "sgl",
        kv_dtype,
        kv_shape,
    )

    use_gpu = need_gpu_interm_buffer(config)

    hidden_dim_size = num_kv_head * head_dim

    if config.use_layerwise:
        raise ValueError("Layerwise connector is not supported yet")
    else:
        sglang_gpu_connector = SGLangGPUConnector(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
        )
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        config,
        metadata,
        sglang_gpu_connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    return engine


class LMCacheConnector:
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        world_size: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
    ):
        kv_dtype = k_pool[0].dtype
        self.lmcache_engine = init_lmcache_engine(
            sgl_config,
            tp_size,
            rank,
            world_size,
            kv_dtype,
        )
        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = rank
        self.world_size = world_size
        self.kvcaches = k_pool + v_pool

    ####################
    # Worker side APIs
    ####################

    def load_kv(
        self, token_ids: torch.Tensor, slot_mapping: torch.Tensor, offset: int = 0
    ) -> None:
        assert isinstance(token_ids, torch.Tensor)
        assert isinstance(slot_mapping, torch.Tensor)
        assert (len(token_ids) - offset) == len(slot_mapping)

        slot_mapping = slot_mapping.cuda()
        load_mask = torch.ones_like(token_ids, dtype=torch.bool)
        load_mask[:offset] = False

        ret_token_mask = self.lmcache_engine.retrieve(
            token_ids,
            mask=load_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=offset,
        )

        num_retrieved_tokens = ret_token_mask.sum().item()

        return num_retrieved_tokens

    def store_kv(
        self, token_ids: torch.Tensor, slot_mapping: torch.Tensor, offset: int = 0
    ) -> None:
        assert isinstance(token_ids, torch.Tensor)
        assert isinstance(slot_mapping, torch.Tensor)
        assert len(token_ids) == len(slot_mapping)

        slot_mapping = slot_mapping.cuda()
        store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=offset,
        )

    def prefetch(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.lmcache_engine.prefetch(tokens, mask)

    def chunk_size(self):
        return self.lmcache_engine.config.chunk_size

    def reset(self):
        self.lmcache_engine.clear()

    def close(self):
        self.lmcache_engine.close()
