import random
import string

import pytest
import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.experimental.memory_management import (HostMemoryAllocator,
                                                    MemoryAllocatorInterface)
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.utils import CacheEngineKey


@pytest.fixture
def mem_allocator():
    size = 1024 * 1024 * 1024  # 1GB
    return HostMemoryAllocator(size)


def random_string(N):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


def get_metadata(kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheEngineMetadata("lmsys/longchat-7b-16k", 1, -1, "vllm",
                                 torch.half, kv_shape)


def generate_random_key() -> CacheEngineKey:
    fmt = random.choice(["vllm", "huggingface"])
    model_name = random_string(10).replace("@", "")
    world_size = 3
    worker_id = random.randint(0, 100)
    chunk_hash = random_string(64)
    return CacheEngineKey(fmt, model_name, world_size, worker_id, chunk_hash)


def generate_memory_object(shape, dtype, allocator: MemoryAllocatorInterface):
    return allocator.allocate(shape, dtype)


def test_basic_put_get(mem_allocator):
    cfg = LMCacheEngineConfig.from_defaults(local_device="cpu",
                                            remote_url=None)
    keys = [generate_random_key() for _ in range(10)]
    values = [
        generate_memory_object((10, 10), torch.float32, mem_allocator)
        for _ in range(10)
    ]
    for i in range(10):
        values[i].tensor.fill_(i)

    storage_manager = StorageManager(cfg, get_metadata(), mem_allocator)

    for key, value in zip(keys, values):
        storage_manager.put(key, value)

    for i in range(10):
        key = keys[i]
        value = values[i]
        retrieved = storage_manager.get(key)
        assert retrieved == value
        assert (retrieved.tensor == i).all()
