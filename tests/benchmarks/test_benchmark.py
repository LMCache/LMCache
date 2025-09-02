# SPDX-License-Identifier: Apache-2.0
# Standard
import random

# Third Party
from utils import (
    create_gpu_connector,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
)
import pytest
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig


# helper functions
def generate_random_slot_mapping(num_blocks, block_size, num_tokens, device):
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    return torch.tensor(slot_mapping, device=device)


# test store 100GB data
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="store")
def test_store_100GB(benchmark, autorelease_v1):
    # model-related metadatas
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    # lmcache and vllm configs
    device = "cuda"
    fmt = "vllm"
    num_tokens = 10000

    num_blocks = 12500
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    num_requests = 80

    # Initialize related modules
    connector = create_gpu_connector(num_heads * head_dim, num_layers)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]

    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
        for _ in range(num_requests)
    ]

    # TODO: Rewrite the config generation to another helper function
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend="cpu")

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    # Run benchmark
    def run_func():
        for t, s in zip(list_tokens, list_slot_mappings):
            engine.store(t, kvcaches=kv_cache, slot_mapping=s)

    benchmark.pedantic(run_func, rounds=1, iterations=1)


# Test retrieve 100GB data (10 rounds, each round 10GB, 100% hit)
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="retrieve")
def test_retrieve_100GB_allhit(benchmark, autorelease_v1):
    # model-related metadatas
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    # lmcache and vllm configs
    device = "cuda"
    fmt = "vllm"
    num_tokens = 10000

    num_blocks = 12500
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    num_requests = 8

    # Initialize related modules
    connector = create_gpu_connector(num_heads * head_dim, num_layers)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]

    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
        for _ in range(num_requests)
    ]

    # TODO: Rewrite the config generation to another helper function
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size, max_local_cpu_size=12
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    for t, s in zip(list_tokens, list_slot_mappings):
        engine.store(t, kvcaches=kv_cache, slot_mapping=s)

    # Run benchmark
    def run_func():
        # TODO: remove the hard-code here
        for i in range(10):
            for t, s in zip(list_tokens, list_slot_mappings):
                engine.retrieve(t, kvcaches=kv_cache, slot_mapping=s)

    benchmark.pedantic(run_func, rounds=1, iterations=1)


# Test lookup 10K * 10 requests, 100% hit
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="lookup")
def test_lookup_10reqs_10Ktokens(benchmark, autorelease_v1):
    # model-related metadatas
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    # lmcache and vllm configs
    device = "cuda"
    fmt = "vllm"
    num_tokens = 10000

    num_blocks = 12500
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    num_requests = 10

    # Initialize related modules
    connector = create_gpu_connector(num_heads * head_dim, num_layers)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]

    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
        for _ in range(num_requests)
    ]

    # TODO: Rewrite the config generation to another helper function
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size, max_local_cpu_size=15
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    for t, s in zip(list_tokens, list_slot_mappings):
        engine.store(t, kvcaches=kv_cache, slot_mapping=s)

    # Run benchmark
    def run_func():
        # TODO: remove the hard-code here
        for t, s in zip(list_tokens, list_slot_mappings):
            engine.lookup(t)

    # Repeat for 10 iterations
    benchmark.pedantic(run_func, rounds=10, iterations=1)
