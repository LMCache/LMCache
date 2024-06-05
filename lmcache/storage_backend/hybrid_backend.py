from typing import Tuple, Optional
import re
import abc
import io
import torch
import redis
import time
import pickle

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.remote_backend import LMCRemoteBackend
from lmcache.storage_backend.local_backend import LMCLocalBackend
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector

logger = init_logger(__name__)

class LMCHybridBackend(LMCBackendInterface):
    """
    A hybrid backend that uses both local and remote backend to store and retrieve data.
    It implements write-through and read-through caching.
    """

    # TODO: LRU eviction policy
    # TODO: async write and read from/to remote backend

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.local_store = LMCLocalBackend(config)
        self.remote_store = LMCRemoteBackend(config)

        # prefetch
        keys = self.remote_store.list()
        nfetched = 0
        logger.info("Found %d keys in remote backend", len(keys))
        logger.debug(f"Metadata is {metadata}")
        for key in keys:
            if key.model_name != metadata.model_name or \
                    key.worker_id != metadata.worker_id or \
                    key.world_size != metadata.world_size:
                continue

            retrived_data = self.remote_store.get(key)
            if retrived_data is not None:
                self.local_store.put(key, retrived_data)
                nfetched += 1
        logger.info("Pre-fetched %d keys from remote backend", nfetched)

    def contains(
            self,
            key: Tuple[str, str],
        ) -> bool:
        return self.local_store.contains(key) or self.remote_store.contains(key)

    def put(
            self,
            key: Tuple[str, str],
            value: torch.Tensor,
        ):
        self.local_store.put(key, value)
        # TODO: considering async write to remote backend
        self.remote_store.put(key, value)

    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[torch.Tensor]:
        value = self.local_store.get(key)
        if value is None:
            value = self.remote_store.get(key)
            if value is not None:
                self.local_store.put(key, value)
        return value
