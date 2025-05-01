from enum import IntEnum
from typing import List, Optional, Union, Tuple

import torch
from sglang.srt.configs.model_config import ModelConfig

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_engine import (LMCacheEngine,
                                               LMCacheEngineBuilder)
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.dram_connector import (SGLangDramNestedConnector,
                                                 SGLangDramNestedConnectorInner)
from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class StoreStatus(IntEnum):
    FAIL = -1
    PREFILLING = 0


class RetrieveStatus(IntEnum):
    FAIL = -1
    PREFILLING = 0


def init_lmcache_engine(
    model_config: ModelConfig,
    rank: int,
    world_size: int,
    tensor_parallel_size: int = 1,
    dram_connector_version: int = 1,
) -> Optional[LMCacheEngine]:
    """
    Initialize the LMCache engine for SGLang.
    :param model_config: The model configuration.
    :param rank: The rank of the current process.
    :param world_size: The total number of processes.
    :param tensor_parallel_size: The size of the tensor parallel group.
    :return: The LMCache engine.
    """

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
    metadata = LMCacheEngineMetadata(model_config.model_path, world_size, rank,
                                     "sglang", kv_dtype, kv_shape)
    hidden_dim_size = num_kv_head * head_size
    if dram_connector_version == 1:
        sglang_dram_connector = SGLangDramNestedConnector(hidden_dim_size,
                                                          num_layer, chunk_size)
    elif dram_connector_version == 2:
        sglang_dram_connector = SGLangDramNestedConnectorInner(hidden_dim_size,
                                                              num_layer, chunk_size)
    else:
        raise ValueError(f"Invalid DRAM connector version: {dram_connector_version}")
    assert isinstance(config, LMCacheEngineConfig), \
        "LMCache experimental configuration is should be passed."
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME, config, metadata, dram_connector=sglang_dram_connector)

    return engine


def get_hash(
        engine: LMCacheEngine,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prefix_hash: Optional[CacheEngineKey] = None) -> List[CacheEngineKey]:
    """
    Get the hash for the given token IDs.
    :param engine: The LMCache engine.
    :param token_ids: The token IDs.
    :param mask: The mask for the token IDs.
    :param prefix_hash: The prefix hash.
    :return: List of CacheEngineKey for the token IDs.
    """
    if prefix_hash is not None:
        prefix_chunk_hash = prefix_hash.chunk_hash
    else:
        prefix_chunk_hash = None
    return engine.get_hash(token_ids, mask=mask, prefix_hash=prefix_chunk_hash)


@_lmcache_nvtx_annotate
def lmcache_store_kv(
    engine: LMCacheEngine,
    token_ids: torch.Tensor,
    kv_caches: Union[torch.Tensor, List[torch.Tensor]],
    prefix_hash: Optional[CacheEngineKey] = None,
) -> Tuple[bool, List[CacheEngineKey]]:
    """
    Store the KV caches in LMCache.
    :param engine: The LMCache engine.
    :param token_ids: The token IDs.
    :param kv_caches: The KV caches to store.
    :param store_status: The status of the store operation.
    :param prefix_hash: The prefix hash.
    :return: List of CacheEngineKey for the stored KV caches.
    """

    store_status = [StoreStatus.FAIL
                    ] * (len(token_ids) // engine.config.chunk_size)
    if prefix_hash is not None:
        prefix_chunk_hash = prefix_hash.chunk_hash
    else:
        prefix_chunk_hash = None
    hash_keys = engine.store(token_ids,
                        mask=None,
                        kvcaches=kv_caches,
                        store_status=store_status,
                        prefix_hash=prefix_chunk_hash)
    return StoreStatus.FAIL not in store_status, hash_keys


@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    engine: LMCacheEngine,
    token_ids: torch.Tensor,
    kv_caches: Union[torch.Tensor, List[torch.Tensor]],
    prefix_hash: Optional[CacheEngineKey] = None,
) -> bool:
    """
    Retrieve the KV caches from LMCache.
    :param engine: The LMCache engine.
    :param token_ids: The token IDs.
    :param kv_caches: The KV caches to be retrieved in.
    :param retrieve_status: The status of the retrieve operation.
    :param prefix_hash: The prefix hash.
    :return: None
    """

    retrieve_status = [RetrieveStatus.FAIL
                           ] * (len(token_ids) // engine.config.chunk_size)
    if prefix_hash is not None:
        prefix_chunk_hash = prefix_hash.chunk_hash
    else:
        prefix_chunk_hash = None
    engine.retrieve(token_ids,
                    mask=None,
                    kvcaches=kv_caches,
                    retrieve_status=retrieve_status,
                    prefix_hash=prefix_chunk_hash)
    return RetrieveStatus.FAIL not in retrieve_status


@_lmcache_nvtx_annotate
def lmcache_store_kv_hash(
    engine: LMCacheEngine,
    hash_: List[CacheEngineKey],
    kv_caches: List[torch.Tensor],
) -> Tuple[bool, List[CacheEngineKey]]:
    """ 
    Store the KV caches in LMCache using existing hash.
    :param engine: The LMCache engine. 
    :param hash_: The CacheEngineKey list for the store block.
    :param kv_caches: The KV caches to store.
    :param store_status: The status of the store operation.
    :return: List of CacheEngineKey for the stored KV caches.
    """
    store_status = [RetrieveStatus.FAIL] * len(hash_)
    hash_keys = engine.hash_store(hash_,
                             kvcaches=kv_caches,
                             store_status=store_status)
    return StoreStatus.FAIL not in store_status, hash_keys


@_lmcache_nvtx_annotate
def lmcache_retrieve_kv_hash(
    engine: LMCacheEngine,
    hash_: List[CacheEngineKey],
    kv_caches: List[torch.Tensor],
) -> bool:
    """
    Retrieve the KV caches from LMCache using existing hash.
    :param engine: The LMCache engine.
    :param hash_: The CacheEngineKey list for the retrieve block.
    :param kv_caches: The KV caches to be retrieved in.
    :param retrieve_status: The status of the retrieve operation.
    :return: None
    """
    retrieve_status = [RetrieveStatus.FAIL] * len(hash_)
    engine.hash_retrieve(hash_,
                         kvcaches=kv_caches,
                         retrieve_status=retrieve_status)
    return RetrieveStatus.FAIL not in retrieve_status
