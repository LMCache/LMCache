from lmcache.integration.sglang.utils import lmcache_get_config
from lmcache.logging import init_logger
from lmcache.experimental.cache_engine import (LMCacheEngine,
                                               LMCacheEngineBuilder)
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.dram_connector import SGLangDramNestedConnector


logger = init_logger(__name__)

def init_lmcache_engine(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine for SGLang."""

    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype,
                                        model_config.dtype)

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)

    # Change current device.
    torch.cuda.device(parallel_config.rank)
    metadata = LMCacheEngineMetadata(model_config.model,
                                     parallel_config.world_size,
                                     parallel_config.rank, "vllm", kv_dtype,
                                     kv_shape)
    hidden_dim_size = num_kv_head * head_size
    sglang_dram_connector = SGLangDramNestedConnector(hidden_dim_size,
                                                      num_layer)
    assert isinstance(config, LMCacheEngineConfig), \
        "LMCache experimental configuration is should be passed."
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME, config, metadata, dram_connector=sglang_dram_connector)

    return engine