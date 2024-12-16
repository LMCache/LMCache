from typing import Dict

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.local_disk_backend import \
    LocalDiskBackend
from lmcache.logging import init_logger

logger = init_logger(__name__)


def CreateStorageBackends(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        dst_device: str = "cuda") -> Dict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: Dict[str, StorageBackendInterface] = {}

    # TODO(Jiayi): The hierarchy is fixed for now
    if config.local_disk and config.max_local_disk_size > 0:
        backend = LocalDiskBackend(config, dst_device)
        backend_name = str(backend)
        storage_backends[backend_name] = backend

    # TODO(Jiayi): Please support other backends
    config.remote_url = None
    config.enable_blending = False
    assert config.remote_url is None, \
        "remote backends are not supported for now"
    assert config.enable_blending is False, \
        "blending is not supported for now"

    return storage_backends
