from typing import Tuple, Optional
import io
import torch

from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector
from lmcache.storage_backend.serde import TorchSerializer, TorchDeserializer
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

# TODO: unit test for the remote connector




class LMCRemoteBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the Redis.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()
        self.existing_keys = set()
        self.connection = CreateConnector(config.remote_url)
        self.serializer = TorchSerializer()
        self.deserializer = TorchDeserializer()

    def _combine_key(
            self,
            key: CacheEngineKey,
        ) -> str:
        """
        Convert the tuple key to a single key
        """
        return key.to_string()

    def _split_key(
            self,
            key: str,
        ) -> CacheEngineKey:
        """
        Split the single key to a tuple key
        """
        return CacheEngineKey.from_string(key)

    def list(self):
        """
        list the remote keys (and also update the 'cached' existing keys set)
        """
        keys = self.connection.list()
        for key in keys:
            self.existing_keys.add(self._split_key(key))
        return [self._split_key(key) for key in keys]

    def contains(
            self, 
            key: CacheEngineKey,
        ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        if key in self.existing_keys:
            return True
        else:
            flag = self.connection.exists(self._combine_key(key))
            if flag:
                self.existing_keys.add(key)
            return flag

    def put(
            self, 
            key: CacheEngineKey,
            kv_chunk: torch.Tensor,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in a single big tensor

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        bs = self.serializer.to_bytes(kv_chunk)
        self.connection.set(self._combine_key(key), bs)

        self.existing_keys.add(key)

    def get(
            self,
            key: CacheEngineKey,
        ) -> Optional[torch.Tensor]:
        """
        Retrive the KV cache chunk (in a single big tensor) by the given key
        """
        if not self.contains(key):
            return None

        bs = self.connection.get(self._combine_key(key))
        if bs is None or len(bs) == 0:
            return None

        return self.deserializer.from_bytes(bs)
