from typing import List, Optional, Union
from enum import Enum
import torch

from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.experimental.cache_engine import (LMCacheEngine,
                                               LMCacheEngineBuilder)
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.dram_connector import SGLangDramNestedConnector
from lmcache.config import LMCacheEngineMetadata

from sglang.srt.configs.model_config import ModelConfig
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class StoreStatus(Enum):
    SUCCESS = 1
    FAILURE = 2

class RetrieveStatus(Enum):
    SUCCESS = 1
    FAILURE = 2

def init_lmcache_engine(
    model_config: ModelConfig,
    rank: int,
    world_size: int,
    tensor_parallel_size: int = 1,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine for SGLang."""

    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()

    kv_dtype = model_config.dtype

    # construct kv shape (for mem pool)
    num_layer = model_config.num_hidden_layers
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(tensor_parallel_size)
    head_size = model_config.head_dim
    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)

    # Change current device.
    torch.cuda.device(rank)
    metadata = LMCacheEngineMetadata(model_config.model_path,
                                     world_size,
                                     rank, "sglang", kv_dtype,
                                     kv_shape)
    hidden_dim_size = num_kv_head * head_size
    sglang_dram_connector = SGLangDramNestedConnector(hidden_dim_size,
                                                      num_layer)
    assert isinstance(config, LMCacheEngineConfig), \
        "LMCache experimental configuration is should be passed."
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME, config, metadata, dram_connector=sglang_dram_connector)

    return engine

def get_hash(engine: LMCacheEngine, token_ids: torch.Tensor) -> str:
    """Get the hash for the given token ids."""
    return engine.get_hash(token_ids)

@_lmcache_nvtx_annotate
def lmcache_store_kv(
    engine: LMCacheEngine,
    token_ids: torch.Tensor,
    kv_caches: Union[torch.Tensor, List[torch.Tensor]],
    store_status: List[StoreStatus],
) -> None:
    engine.store(token_ids, mask=None, kvcaches=kv_caches, store_status=store_status)

@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    engine: LMCacheEngine,
    token_ids: torch.Tensor,
    kv_caches: Union[torch.Tensor, List[torch.Tensor]],
    retrieve_status: List[RetrieveStatus],
) -> None:
    engine.retrieve(token_ids, mask=None, kvcaches=kv_caches, retrieve_status=retrieve_status)

@_lmcache_nvtx_annotate
def lmcache_retrieve_kv_hash(
    engine: LMCacheEngine,
    hash: List[str],
    kv_caches: List[torch.Tensor],
    retrieve_status: List[RetrieveStatus],
) -> None:
    pass

@_lmcache_nvtx_annotate
def lmcache_retrieve_kv_hash(
    engine: LMCacheEngine,
    hash_: List[str],
    kv_caches: List[torch.Tensor],
    retrieve_status: List[RetrieveStatus],
) -> None:
    return engine.hash_retrieve(hash_, kvcaches=kv_caches, retrieve_status=retrieve_status)
