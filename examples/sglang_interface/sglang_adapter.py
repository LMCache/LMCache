import os
import torch

from lmcache.integration.sglang.utils import ENGINE_NAME
from lmcache.integration.sglang.sglang_adapter import init_lmcache_engine, lmcache_store_kv, lmcache_retrieve_kv, get_hash, lmcache_retrieve_kv_hash, StoreStatus, RetrieveStatus
from lmcache.experimental.cache_engine import LMCacheEngineBuilder

from sglang.srt.configs.model_config import ModelConfig

os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"


def test():
    model_config = ModelConfig(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_override_args="{}")
    rank = 0
    world_size = 1
    tensor_parallel_size = 1
    kv_pool_size = 10000
    seq_len = 512
    engine = init_lmcache_engine(model_config, rank, world_size,
                                 tensor_parallel_size)

    # Generate a random tensor to simulate tokens, 512 tokens
    tokens = torch.randint(0, 10000, (seq_len, )).cuda()

    # Generate a kv cache pool same as hirachical kv cache
    kv_cache = torch.randn(2, model_config.num_hidden_layers, kv_pool_size,
                           model_config.get_num_kv_heads(tensor_parallel_size),
                           model_config.head_dim)

    # Random Indices for tokens
    indices = torch.tensor([i for i in range(seq_len)])

    load_status = torch.full((seq_len // engine.config.chunk_size, ),
                             StoreStatus.FAIL,
                             dtype=torch.int32)
    hash_keys = lmcache_store_kv(engine, tokens, kv_cache[:, :, indices],
                                 load_status)
    print("Failure in stroing kv cache in LMCache:",
          torch.any(load_status == StoreStatus.FAIL))
    print("KV cache stored in LMCache.")

    buffer = torch.zeros(2, model_config.num_hidden_layers, seq_len,
                         model_config.get_num_kv_heads(tensor_parallel_size),
                         model_config.head_dim)
    retrieve_status = torch.full((seq_len // engine.config.chunk_size, ),
                                 RetrieveStatus.FAIL,
                                 dtype=torch.int32)
    lmcache_retrieve_kv(engine, tokens, buffer, retrieve_status)
    print("Failure in retrieving kv cache from LMCache:",
          torch.any(retrieve_status == RetrieveStatus.FAIL))
    print("KV cache retrieved from LMCache.")

    # Assert the retrieved kv cache is close to the stored one
    if torch.allclose(buffer, kv_cache[:, :, indices], atol=1e-1):
        print("KV cache retrieved successfully matches the stored one.")
    else:
        print("KV cache retrieved does not match the stored one.")

    hash_ = get_hash(engine, tokens)
    for i, entry in enumerate(zip(hash_keys, hash_)):
        if entry[0] != entry[1]:
            print(f"Hash mismatch at index {i}: {entry[0]} != {entry[1]}")

    buffer = torch.zeros(2, model_config.num_hidden_layers, seq_len,
                         model_config.get_num_kv_heads(tensor_parallel_size),
                         model_config.head_dim)
    retrieve_status = torch.full((seq_len // engine.config.chunk_size, ),
                                 RetrieveStatus.FAIL,
                                 dtype=torch.int32)
    lmcache_retrieve_kv_hash(engine, hash_, buffer, retrieve_status)
    print("Failure in retrieving kv cache from LMCache using hash:",
          torch.any(retrieve_status == RetrieveStatus.FAIL))
    print("KV cache retrieved from LMCache using hash.")

    if torch.allclose(buffer, kv_cache[:, :, indices], atol=1e-1):
        print("KV cache retrieved successfully matches the stored one.")
    else:
        print("KV cache retrieved does not match the stored one.")

    # Begin the prefix operation
    prefix_hash = hash_[-1]
    suffix_tokens = torch.randint(0, 10000, (seq_len, )).cuda()
    suffix_indices = torch.tensor([i + 2048 for i in range(seq_len)])
    suffix_hash = get_hash(engine, suffix_tokens, prefix_hash=prefix_hash)

    suffix_hash_keys = lmcache_store_kv(engine, suffix_tokens,
                                        kv_cache[:, :, suffix_indices],
                                        load_status, prefix_hash)
    print("Failure in stroing kv cache in LMCache:",
          torch.any(load_status == StoreStatus.FAIL))
    print("Suffix KV cache stored in LMCache.")

    for i, entry in enumerate(zip(suffix_hash_keys, suffix_hash)):
        if entry[0] != entry[1]:
            print(f"Hash mismatch at index {i}: {entry[0]} != {entry[1]}")

    buffer = torch.zeros(2, model_config.num_hidden_layers, seq_len,
                         model_config.get_num_kv_heads(tensor_parallel_size),
                         model_config.head_dim)
    retrieve_status = torch.full((seq_len // engine.config.chunk_size, ),
                                 RetrieveStatus.FAIL,
                                 dtype=torch.int32)
    lmcache_retrieve_kv(engine, suffix_tokens, buffer, retrieve_status,
                        prefix_hash)
    print("Failure in retrieving kv cache from LMCache:",
          torch.any(retrieve_status == RetrieveStatus.FAIL))

    print("Suffix KV cache retrieved from LMCache.")
    # Assert the retrieved kv cache is close to the stored one
    if torch.allclose(buffer, kv_cache[:, :, suffix_indices], atol=1e-1):
        print("Suffix KV cache retrieved successfully matches the stored one.")
    else:
        print("Suffix KV cache retrieved does not match the stored one.")

    buffer.zero_()
    retrieve_status = torch.full((seq_len // engine.config.chunk_size, ),
                                 RetrieveStatus.FAIL,
                                 dtype=torch.int32)
    lmcache_retrieve_kv_hash(engine, suffix_hash, buffer, retrieve_status)
    print("Failure in retrieving kv cache from LMCache using hash:",
          torch.any(retrieve_status == RetrieveStatus.FAIL))
    print("Suffix KV cache retrieved from LMCache using hash.")

    # Assert the retrieved kv cache is close to the stored one
    if torch.allclose(buffer, kv_cache[:, :, suffix_indices], atol=1e-1):
        print("Suffix KV cache retrieved successfully matches the stored one.")
    else:
        print("Suffix KV cache retrieved does not match the stored one.")

    # Overall combination
    tokens = torch.cat((tokens, suffix_tokens))
    indices = torch.cat((indices, suffix_indices))
    buffer = torch.zeros(2, model_config.num_hidden_layers, tokens.shape[0],
                         model_config.get_num_kv_heads(tensor_parallel_size),
                         model_config.head_dim)
    retrieve_status = torch.full(
        (tokens.shape[0] // engine.config.chunk_size, ),
        RetrieveStatus.FAIL,
        dtype=torch.int32)
    lmcache_retrieve_kv(engine, tokens, buffer, retrieve_status)
    print("Failure in retrieving kv cache from LMCache:",
          torch.any(retrieve_status == RetrieveStatus.FAIL))
    print("Overall KV cache retrieved from LMCache.")

    # Assert the retrieved kv cache is close to the stored one
    if torch.allclose(buffer, kv_cache[:, :, indices], atol=1e-1):
        print(
            "Overall KV cache retrieved successfully matches the stored one.")
    else:
        print("Overall KV cache retrieved does not match the stored one.")

    # Clean up
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    test()
