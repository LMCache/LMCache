import os
import torch

from lmcache.integration.sglang.utils import ENGINE_NAME
from lmcache.integration.sglang.sglang_adapter import init_lmcache_engine, lmcache_store_kv, lmcache_retrieve_kv, get_hash, lmcache_retrieve_kv_hash
from lmcache.experimental.cache_engine import LMCacheEngineBuilder

from sglang.srt.configs.model_config import ModelConfig

os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"

def test():
    model_config = ModelConfig(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct", model_override_args="{}"
    )
    rank = 0
    world_size = 1
    tensor_parallel_size = 1
    kv_pool_size = 10000
    seq_len = 512
    engine = init_lmcache_engine(model_config, rank, world_size, tensor_parallel_size)
    
    # Generate a random tensor to simulate tokens, 512 tokens
    tokens = torch.randint(0, 10000, (seq_len,)).cuda()
    
    # Generate a kv cache pool same as hirachical kv cache
    kv_cache = torch.randn(2, model_config.num_hidden_layers, kv_pool_size, model_config.get_num_kv_heads(tensor_parallel_size), model_config.head_dim)
    
    # Store the kv cache in LMCache
    indices = torch.tensor([i for i in range(seq_len)])
    
    load_status = torch.zeros(seq_len, dtype=torch.int32)
    lmcache_store_kv(engine, tokens, kv_cache[:, :, indices], load_status)
    print("KV cache stored in LMCache.")
    
    buffer = torch.zeros(2, model_config.num_hidden_layers, seq_len, model_config.get_num_kv_heads(tensor_parallel_size), model_config.head_dim)
    retrieve_status = torch.zeros(seq_len, dtype=torch.int32)
    lmcache_retrieve_kv(engine, tokens, buffer, retrieve_status)
    print("KV cache retrieved from LMCache.")
    
    # Assert the retrieved kv cache is close to the stored one
    print(torch.allclose(buffer, kv_cache[:, :, indices], atol=1e-1))
    print("KV cache retrieved successfully matches the stored one.")
    
    hash_ = get_hash(engine, tokens)
    buffer = torch.zeros(2, model_config.num_hidden_layers, seq_len, model_config.get_num_kv_heads(tensor_parallel_size), model_config.head_dim)
    retrieve_status = torch.zeros(seq_len // engine.config.chunk_size, dtype=torch.int32)
    lmcache_retrieve_kv_hash(engine, hash_, buffer, retrieve_status)
    print("KV cache retrieved from LMCache using hash.")
    
    print(torch.allclose(buffer, kv_cache[:, :, indices], atol=1e-1))
    print("KV cache retrieved successfully matches the stored one.")
       
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    
if __name__ == "__main__":
    test()