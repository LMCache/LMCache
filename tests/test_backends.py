import pytest
from pathlib import Path
import time
import torch
import random
import string

from lmcache.storage_backend import CreateStorageBackend
from lmcache.storage_backend.local_backend import LMCLocalBackend, LMCLocalDiskBackend
from lmcache.storage_backend.remote_backend import LMCRemoteBackend
from lmcache.storage_backend.hybrid_backend import LMCHybridBackend
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey

from lmcache.logging import init_logger
logger = init_logger(__name__)

LMSEVER_URL = "lm://localhost:65000"
REDIS_URL = "redis://localhost:6379"

def random_string(N):
    return ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))

def generate_random_key() -> CacheEngineKey:
    fmt = random.choice(["vllm", "huggingface"])
    model_name = random_string(10).replace("@", "")
    world_size = 3
    worker_id = random.randint(0, 100)
    chunk_hash = random_string(64)
    return CacheEngineKey(fmt, model_name, world_size, worker_id, chunk_hash)

def get_config(t):
    match t:
        case "local":
            return LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url = None)
        case "remote":
            return LMCacheEngineConfig.from_defaults(local_device = None, remote_url="lm://localhost:65000")
        case "hybrid":
            return LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url="lm://localhost:65000")
        case "hybrid_pipelined":
            return LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url="lm://localhost:65000", pipelined_backend=True)
        case _:
            raise ValueError(f"Testbed internal error: Unknown config type: {t}")

def get_metadata():
    return LMCacheEngineMetadata("lmsys/longchat-7b-16k", 1, -1, "vllm")

@pytest.mark.parametrize("lmserver_process", ["cpu", "remote_disk/"], indirect=True)
def test_creation(autorelease, lmserver_process):
    config_local = LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url = None)
    config_remote = LMCacheEngineConfig.from_defaults(local_device = None, remote_url="lm://localhost:65000")
    config_hybrid = LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url="lm://localhost:65000")
    metadata = get_metadata()
    
    backend_local = autorelease(CreateStorageBackend(config_local, get_metadata()))
    backend_remote = autorelease(CreateStorageBackend(config_remote, get_metadata()))
    backend_hybrid = autorelease(CreateStorageBackend(config_hybrid, get_metadata()))

    assert isinstance(backend_local, LMCLocalBackend)
    assert isinstance(backend_remote, LMCRemoteBackend)
    assert isinstance(backend_hybrid, LMCHybridBackend)

    config_fail = LMCacheEngineConfig.from_defaults(local_device = None, remote_url = None)
    with pytest.raises(ValueError):
        backend_fail = CreateStorageBackend(config_fail, get_metadata())

@pytest.mark.parametrize("lmserver_process", ["cpu"], indirect=True)
def test_creation_from_file(autorelease, lmserver_process):
    BASE_DIR = Path(__file__).parent

    config_local = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_creation_from_file/local.yaml")
    config_disk = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_creation_from_file/disk.yaml")
    config_remote = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_creation_from_file/remote.yaml")
    config_hybrid = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_creation_from_file/hybrid.yaml")
    metadata = get_metadata()
    
    backend_local = autorelease(CreateStorageBackend(config_local, get_metadata()))
    backend_disk = autorelease(CreateStorageBackend(config_disk, get_metadata()))
    backend_remote = autorelease(CreateStorageBackend(config_remote, get_metadata()))
    backend_hybrid = autorelease(CreateStorageBackend(config_hybrid, get_metadata()))

    assert isinstance(backend_local, LMCLocalBackend)
    assert isinstance(backend_disk, LMCLocalDiskBackend)
    assert isinstance(backend_remote, LMCRemoteBackend)
    assert isinstance(backend_hybrid, LMCHybridBackend)

    config_fail = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_creation_from_file/fail.yaml")
    with pytest.raises(ValueError):
        backend_fail = CreateStorageBackend(config_fail, get_metadata())

@pytest.mark.parametrize("backend_type", ["local", "remote", "hybrid", "hybrid_pipelined"])
@pytest.mark.parametrize("lmserver_process", ["cpu", "remote_disk/"], indirect=True)
def test_backends(backend_type, autorelease, lmserver_process):
    config = get_config(backend_type) 
    metadata = get_metadata()
    backend = autorelease(CreateStorageBackend(config, metadata))
    
    N = 10
    keys = [generate_random_key() for i in range(N)]
    random_tensors = [torch.rand((16, 2, 128, 4, 128)) for i in range(N)]
    for key, value in zip(keys, random_tensors):
        backend.put(key, value)

    for key, value in zip(keys, random_tensors):
        assert backend.contains(key)
        retrived = backend.get(key)
        assert retrived.shape == value.shape
        if config.remote_serde == "torch":
            assert torch.equal(value, retrived.to(value.device))

@pytest.mark.parametrize("backend_type", ["local", "remote", "hybrid"])
@pytest.mark.parametrize("lmserver_process", ["cpu", "remote_disk/"], indirect=True)
def test_nonblocking_put(backend_type, autorelease, lmserver_process):
    config = get_config(backend_type) 
    metadata = get_metadata()
    backend = autorelease(CreateStorageBackend(config, metadata))
    
    N = 10
    keys = [generate_random_key() for i in range(N)]
    random_tensors = [torch.rand((16, 2, 128, 4, 128)) for i in range(N)]

    for key, value in zip(keys, random_tensors):
        start = time.perf_counter()
        backend.put(key, value, blocking=False)
        end = time.perf_counter()
        elapsed = end - start
        # TODO: `end = time.perf_counter()` may not be instantly scheduled after backed.put()
        #       So we cannot assert on the elapsed time. This should be fixed in the future
        #assert elapsed < 0.005
    
    time.sleep(5)
    for key, value in zip(keys, random_tensors):
        assert backend.contains(key)
        retrived = backend.get(key)
        assert retrived.shape == value.shape
        if config.remote_serde == "torch":
            assert torch.equal(value, retrived.to(value.device))

@pytest.mark.parametrize("lmserver_process", ["cpu", "remote_disk/"], indirect=True)
def test_restart(autorelease, lmserver_process):
    config = get_config("hybrid") #LMCacheEngineConfig.from_defaults(local_device = "cuda", remote_url = None)
    metadata = get_metadata()
    backend = autorelease(CreateStorageBackend(config, metadata))
    
    N = 10
    keys = [generate_random_key() for i in range(N)]
    random_tensors = [torch.rand((1000, 1000)) for i in range(N)]
    for key, value in zip(keys, random_tensors):
        backend.put(key, value)


    new_backend = autorelease(CreateStorageBackend(config, metadata))
    # it should be able to automatically fetch existing keys
    for key, value in zip(keys, random_tensors):
        assert backend.contains(key)
        retrived = backend.get(key)
        assert value.shape == retrived.shape
        if config.remote_serde == "torch":
            assert (value == retrived.to(value.device)).all()
