import time
import os
import torch
import argparse

from sglang.srt.configs.model_config import ModelConfig


from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.sglang.sglang_adapter import (
    RetrieveStatus,
    StoreStatus,
    get_hash,
    init_lmcache_engine,
    lmcache_retrieve_kv,
    lmcache_retrieve_kv_hash,
    lmcache_store_kv,
    lmcache_store_kv_hash,
)
from lmcache.integration.sglang.utils import ENGINE_NAME

import logging

logger = logging.getLogger("Benchmark")
logger.setLevel(logging.INFO)


os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"


def benchmark(args):
    model_config = ModelConfig(
        model_path=args.model_path, model_override_args="{}"
    )
    rank = 0
    world_size = 1
    tensor_parallel_size = 1
    kv_pool_size = args.kv_pool_size
    engine = init_lmcache_engine(model_config, rank, world_size, tensor_parallel_size, args.dram_connector_version)
    logger.info(f"chunk_size: {engine.config.chunk_size}")

    if args.dram_connector_version == 1:
        mem_pool = torch.randn(
            2,
            model_config.num_hidden_layers,
            kv_pool_size,
            model_config.get_num_kv_heads(tensor_parallel_size),
            model_config.head_dim,
        )
    elif args.dram_connector_version == 2:
        mem_pool = torch.randn(
            2,
            kv_pool_size,
            model_config.num_hidden_layers,
            model_config.get_num_kv_heads(tensor_parallel_size),
            model_config.head_dim,
        )
        
    seq_len = args.seq_len
    tokens = torch.load("token.pt")
    if args.dram_connector_version == 1:
        buffer = torch.zeros(
            2,
            model_config.num_hidden_layers,
            seq_len,
            model_config.get_num_kv_heads(tensor_parallel_size),
                model_config.head_dim,
        )
    elif args.dram_connector_version == 2:
        buffer = torch.zeros(
            2,
            seq_len,
            model_config.num_hidden_layers,
            model_config.get_num_kv_heads(tensor_parallel_size),
            model_config.head_dim,
        )
    
    print(tokens)
    
    load_time = list()
    for i in range(args.seq_num):
        tic = time.perf_counter()
        success = lmcache_retrieve_kv(engine, tokens, buffer)
        load_time.append(time.perf_counter() - tic)
        if not success:
            logger.warning("Failed to retrieve kv cache from LMCache")
        tokens = tokens + 1
    
    kvsize = (
        2
        * model_config.num_hidden_layers
        * seq_len
        * model_config.get_num_kv_heads(tensor_parallel_size)
        * model_config.head_dim
    ) * mem_pool.element_size()
    
    avg_load_time = sum(load_time) / len(load_time)
    warmup_time = sum(load_time[:50]) / 50

    logger.info(f"load throughput(cold cache): {kvsize / avg_load_time / 1e6} MB/s,{avg_load_time}")
    logger.info(f"load throughput(system warmup): {kvsize / warmup_time / 1e6} MB/s,{warmup_time}")  
    logger.info("Benchmark finished.")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--kv_pool_size", type=int, default=100000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--seq_num", type=int, default=100)
    parser.add_argument("--dram_connector_version", type=int, default=1)
    args = parser.parse_args()
    benchmark(args)
