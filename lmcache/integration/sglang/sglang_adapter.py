# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, List, Optional
import uuid

# Third Party
from sglang.srt.configs.model_config import ModelConfig
import torch
import torch.distributed as dist

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangGPUConnector,
    SGLangLayerwiseGPUConnector,
)

logger = init_logger(__name__)


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.enable_pd:
        return False
    else:
        return True


@dataclass
class StoreMetadata:
    last_node: Any
    token_ids: List[int]
    kv_indices: torch.Tensor
    offset: int


@dataclass
class LoadMetadata:
    token_ids: List[int]
    slot_mapping: torch.Tensor
    offset: int


def init_lmcache_engine(
    model_config: ModelConfig,
    tp_size: int,
    rank: int,
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
        tp_size,
        rank,
        "sgl",
        kv_dtype,
        kv_shape,
    )

    use_gpu = need_gpu_interm_buffer(config)

    hidden_dim_size = num_kv_head * head_dim

    gpu_connector: GPUConnectorInterface

    if config.use_layerwise:
        gpu_connector = SGLangLayerwiseGPUConnector(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
        )
    else:
        gpu_connector = SGLangGPUConnector(
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
        gpu_connector,
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
        self.num_layer = sgl_config.num_hidden_layers

    ####################
    # Worker side APIs
    ####################

    def load_kv(self, load_metadata: LoadMetadata) -> int:
        token_ids = torch.tensor(load_metadata.token_ids, dtype=torch.int64).cuda()
        slot_mapping = load_metadata.slot_mapping.cuda()
        offset = load_metadata.offset

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

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        token_ids = torch.tensor(store_metadata.token_ids, dtype=torch.int64).cuda()
        slot_mapping = store_metadata.kv_indices.to(torch.int64).cuda()
        offset = store_metadata.offset

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
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(sgl_config, tp_size, rank, k_pool, v_pool)
        self._lmcache_chunk_size = self.lmcache_engine.config.chunk_size
        self.layerwise_retrievers: List[Any] = []
        self.layer_load_layer: List[int] = []
        self.kvcaches = [k_pool, v_pool]
        self.tp_group = tp_group
        self.lookup_id_list: List[str] = []

    @torch.no_grad()
    def global_min_tokens(
        self, local_tokens: int, tp_group: dist.ProcessGroup, device: torch.device
    ):
        t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp_group)
        return int(t.item())

    def load_kv_layerwise(self, layer_id: int) -> None:
        if len(self.layerwise_retrievers) == 0:
            return

        indices_to_remove = []
        for i in range(len(self.layerwise_retrievers)):
            if self.layer_load_layer[i] == layer_id + 1:
                next(self.layerwise_retrievers[i])
                self.layer_load_layer[i] += 1
                if self.layer_load_layer[i] == self.sgl_config.num_hidden_layers:
                    indices_to_remove.append(i)

        for i in sorted(indices_to_remove, reverse=True):
            del self.layerwise_retrievers[i]
            del self.layer_load_layer[i]
            self.lmcache_engine.lookup_unpin([self.lookup_id_list[i]])
            del self.lookup_id_list[i]

        return

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        token_ids = torch.tensor(load_metadata.token_ids, dtype=torch.int64).cuda()
        slot_mapping = load_metadata.slot_mapping.cuda()
        offset = load_metadata.offset

        assert self.lmcache_engine is not None

        load_mask = torch.ones_like(token_ids, dtype=torch.bool)
        load_mask[:offset] = False

        lookup_id = str(uuid.uuid4())
        retrieve_token_num = self.lmcache_engine.lookup(
            token_ids,
            lookup_id=lookup_id,
            pin=True,
        )

        retrieve_token_num = self.global_min_tokens(
            retrieve_token_num, self.tp_group, self.rank
        )

        layerwise_retriever = self.lmcache_engine.retrieve_layer(
            token_ids[:retrieve_token_num],
            mask=load_mask[:retrieve_token_num],
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping[:retrieve_token_num],
            sync=False,
        )

        next(layerwise_retriever)
        # Load First Layer
        next(layerwise_retriever)

        if retrieve_token_num is None:
            return 0

        self.layerwise_retrievers.append(layerwise_retriever)
        self.layer_load_layer.append(1)

        self.lookup_id_list.append(lookup_id)

        return retrieve_token_num - offset

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        slot_mapping = store_metadata.kv_indices.to(torch.int64).cuda()
        token_ids = torch.tensor(store_metadata.token_ids, dtype=torch.int64).cuda()
        store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        lookup_id = str(uuid.uuid4())
        self.lmcache_engine.lookup(token_ids, lookup_id=lookup_id, pin=True)

        layerwise_storer = self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=store_metadata.offset,
            sync=False,
        )
        next(layerwise_storer)
        for _ in range(self.sgl_config.num_hidden_layers):
            next(layerwise_storer)

        self.lmcache_engine.lookup_unpin([lookup_id])
