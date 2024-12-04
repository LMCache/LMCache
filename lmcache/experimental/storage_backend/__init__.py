from typing import Dict

import torch

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.config import LMCacheEngineMetadata

from lmcache.logging import init_logger
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface

from lmcache.experimental.storage_backend.local_disk_backend import \
    LocalDiskBackend


logger = init_logger(__name__)


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    dst_device: str = "cuda"
    ) -> Dict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"
    
    storage_backends: Dict[str, StorageBackendInterface] = {}
    
    # TODO(Jiayi): The heirarchy is fixed for now
    
    if config.local_disk:
        backend = LocalDiskBackend(config, dst_device)
        backend_name = str(backend)
        storage_backends[backend_name] = backend
    
    # TODO(Jiayi): please modify the following checks if
    # the corresponding backends are supported
    assert config.remote_url is None, \
        "remote backends are not supported for now"
    assert config.enable_blending is False, \
        "blending is not suppoerted for now"
    
    return storage_backends
