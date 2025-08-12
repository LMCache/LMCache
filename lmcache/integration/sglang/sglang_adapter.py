# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import (
    SGLangGPUConnector,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ForwardBatch
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.lmc_radix_cache import StoreMetadata


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
    kv_dtype: torch.dtype,
) -> Optional[LMCacheEngine]:
    """
    TODO: ADD COMMENTS
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

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
        tp_size,
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
        ENGINE_NAME, config, metadata, sglang_gpu_connector
    )

    return engine


class LMCacheConnector:
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
    ):
        kv_dtype = k_pool[0].dtype
        self.lmcache_engine = init_lmcache_engine(
            sgl_config,
            tp_size,
            rank,
            kv_dtype,
        )
        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = rank
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

class LMCacheLayerwiseConnector(LMCacheConnector):
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
    ):
        super().__init__(sgl_config, tp_size, rank, k_pool, v_pool)
        self.current_layer = 0
        self._lmcache_chunk_size = self.lmcache_engine.config.chunk_size
        self.layerwise_retrievers = []

    def load_kv_layerwise(self, forward_batch: ForwardBatch) -> None:
        self.current_layer = 0

        self.lmcache_engine.post_init(kvcaches=self.kvcaches)

        self.layerwise_retrievers = []

        for idx, request in enumerate(forward_batch.requests):
            if request.load_spec is None:
                continue
            last_idx = idx

        for idx, request in enumerate(forward_batch.requests):
            tokens = request.input_ids
            slot_mapping = request.token_pool_mapping

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            if idx == last_idx:
                sync = True
            else:
                sync = False

            layerwise_retriever = self.lmcache_engine.retrieve_layer(
                tokens[:masked_token_count],
                token_mask[:masked_token_count],
                kvcaches=self.kvcaches,
                slot_mapping=slot_mapping[:masked_token_count],
                sync=sync,
            )
            next(layerwise_retriever)
            self.layerwise_retrievers.append(layerwise_retriever)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")
        return

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        slot_mapping = store_metadata.kv_indices.cuda()
        token_ids = store_metadata.token_ids
        store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=store_metadata.offset,
        )
        