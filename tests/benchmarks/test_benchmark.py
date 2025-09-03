# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import partial
import os
import random
import tempfile
import time

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


@pytest.fixture
def create_config():
    """
    backend can be:
    - cpu
    - disk
    - fsconnector
    """

    def make_config(backend, size, **kwargs):
        match backend:
            case "cpu":
                return LMCacheEngineConfig.from_defaults(
                    local_cpu=True,
                    max_local_cpu_size=size,
                    extra_config={"force_store_wait": False},
                )
            case "disk":
                assert "path" in kwargs, "'path' is missing for disk backend"
                return LMCacheEngineConfig.from_defaults(
                    local_cpu=False,
                    max_local_cpu_size=size,
                    local_disk=kwargs["path"],
                    max_local_disk_size=size,
                    extra_config={"force_store_wait": False},
                )
            case "fsconnector":
                assert "path" in kwargs, "'path' is missing for fsconnector"
                p = kwargs["path"]
                return LMCacheEngineConfig.from_defaults(
                    local_cpu=False,
                    max_local_cpu_size=size,
                    remote_url=f"fs://host:0/{p}/",
                    remote_serde="naive",
                    extra_config={"force_store_wait": False},
                )
            case _:
                print(f"Error: unknown backend: {backend}")
                print("Supported backends: 'cpu', 'disk', and 'fsconnector'")
                raise ValueError(f"Unknown backend: {backend}")

    homedir = os.environ.get("HOME", "/tmp")
    with tempfile.TemporaryDirectory(
        dir=homedir, ignore_cleanup_errors=True
    ) as temp_dir:
        print("Temp dir is:", temp_dir)
        yield partial(make_config, path=temp_dir)


# test store 100GB data
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="store")
@pytest.mark.parametrize("backend", ["cpu", "disk", "fsconnector"])
def test_store_100GB(benchmark, backend, create_config, autorelease_v1):
    # model-related metadatas
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    # lmcache and vllm configs
    device = "cuda"
    fmt = "vllm"
    num_tokens = 10000

    num_blocks = 5000
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    # - single request has 1.25GB KV, 8 requests has 10GB
    # so we want to do 10 rounds
    num_requests = 8
    num_repeats = 10

    # Initialize related modules
    connector = create_gpu_connector(num_heads * head_dim, num_layers)
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    # TODO: Rewrite the config generation to another helper function
    # cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend="cpu")
    cache_size = 15  # Allocate 15 GB KV cache buffer
    cfg = create_config(backend, cache_size)

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
    def run_func(tokens, slot_mappings):
        for t, s in zip(tokens, slot_mappings):
            engine.store(t, kvcaches=kv_cache, slot_mapping=s)

        # Wait for all tokens are being stored
        timeout = 60
        start = time.time()
        while time.time() - start < timeout:
            ready = all([engine.lookup(t) == len(t) for t in tokens])
            if ready:
                return
            else:
                time.sleep(0.05)
        raise TimeoutError(f"Store operation haven't finished in {timeout} seconds")

    def setup():
        list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]

        list_slot_mappings = [
            generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
            for _ in range(num_requests)
        ]
        return (list_tokens, list_slot_mappings), {}

    benchmark.pedantic(run_func, setup=setup, rounds=num_repeats, iterations=1)


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

    num_blocks = 5000
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    # - Single request has 1.25 GB KV, 8 requests will use 10GB,
    # so num repeats should be 10 to achieve 100 GB access
    num_requests = 8
    num_repeats = 10

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
        for i in range(num_repeats):
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

    num_blocks = 5000
    block_size = 16

    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # Test configs
    num_requests = 10
    num_repeats = 1

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
        for i in range(num_repeats):
            for t, s in zip(list_tokens, list_slot_mappings):
                engine.lookup(t)

    # Repeat for 10 iterations
    benchmark.pedantic(run_func, rounds=10, iterations=1)
