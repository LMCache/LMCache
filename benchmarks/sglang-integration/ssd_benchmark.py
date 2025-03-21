import argparse
import os
import time

import torch
from sglang.srt.configs.model_config import ModelConfig

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.sglang.sglang_adapter import (
    get_hash, init_lmcache_engine, lmcache_retrieve_kv,
    lmcache_retrieve_kv_hash, lmcache_store_kv)
from lmcache.integration.sglang.utils import ENGINE_NAME

os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"


def benchmark(args):
    print("Initializing LMCache engine...")
    model_config = ModelConfig(model_path=args.model_path,
                               model_override_args="{}")

    engine = init_lmcache_engine(model_config, args.rank, args.world_size,
                                 args.tensor_parallel_size)

    assert args.seq_len % args.chunk_size == 0, \
        "seq_len should be divisible by chunk_size"

    # Generate random tokens
    token_pool = [
        torch.randint(0, 10000, (args.seq_len, )) for _ in range(args.seq_num)
    ]

    # Initialize KV Cache (Random values for benchmarking)
    kv_cache = torch.randn(
        2, model_config.num_hidden_layers, args.kv_pool_size,
        model_config.get_num_kv_heads(args.tensor_parallel_size),
        model_config.head_dim)

    # Generate indices for KV cache
    indices = [
        torch.randint(0, args.kv_pool_size, (args.seq_len, ))
        for _ in range(args.seq_num)
    ]
    indices = torch.stack(indices, dim=0)

    # Initialize status tensors
    load_status = torch.zeros(args.seq_num, args.seq_len, dtype=torch.int32)
    retrieve_status = torch.zeros(args.seq_num,
                                  args.seq_len,
                                  dtype=torch.int32)

    # -----------------------------------------------
    # Measure Storage Latency
    # -----------------------------------------------
    start_time = time.time()
    for i in range(args.seq_num):
        lmcache_store_kv(engine, token_pool[i], kv_cache[:, :, indices[i]],
                         load_status[i])
    store_latency = (time.time() - start_time) * 1000
    print(f"KV-cache storage latency:  \
        {store_latency / args.seq_num:.3f} ms per sequence")
    time.sleep(2)

    # -----------------------------------------------
    # Measure Retrieval Latency
    # -----------------------------------------------

    start_time = time.time()
    for i in range(args.seq_num):
        lmcache_retrieve_kv(engine, token_pool[i], kv_cache[:, :, indices[i]],
                            retrieve_status[i])
    retrieve_latency = (time.time() - start_time) * 1000
    print(f"KV-cache retrieval latency: \
        {retrieve_latency / args.seq_num:.3f} ms per sequence")

    # -----------------------------------------------
    # Measure Retrieval Latency Using Hash
    # -----------------------------------------------
    hash_pool = [get_hash(engine, token_pool[i]) for i in range(args.seq_num)]

    start_time = time.time()
    for i in range(args.seq_num):
        lmcache_retrieve_kv_hash(engine, hash_pool[i], kv_cache[:, :,
                                                                indices[i]],
                                 retrieve_status[i])
    retrieve_latency_hash = (time.time() - start_time) * 1000
    print(f"KV-cache retrieval latency (hash-based): \
        {retrieve_latency_hash / args.seq_num:.3f} ms per sequence")

    # -----------------------------------------------
    # Compute Throughput
    # -----------------------------------------------
    throughput = args.seq_num / (retrieve_latency / 1000)
    throughput_hash = args.seq_num / (retrieve_latency_hash / 1000)

    print(f"KV-cache retrieval throughput: {throughput:.2f} sequences/s")
    print(f"KV-cache retrieval throughput (hash-based): \
        {throughput_hash:.2f} sequences/s")

    # Clean up
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("LMCache engine destroyed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LMCache")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to the model")
    parser.add_argument("--kv_pool_size",
                        type=int,
                        default=10000,
                        help="Size of the KV pool")
    parser.add_argument("--seq_len",
                        type=int,
                        default=512,
                        help="Sequence length")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="Rank of the process")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="Total number of processes")
    parser.add_argument("--tensor_parallel_size",
                        type=int,
                        default=1,
                        help="Tensor parallel size")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=1,
                        help="Chunk size for storing and retrieving KV cache")
    parser.add_argument("--seq_num",
                        type=int,
                        default=10,
                        help="Number of sequences to benchmark")
    args = parser.parse_args()

    benchmark(args)
