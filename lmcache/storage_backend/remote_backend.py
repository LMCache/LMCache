from typing import Tuple, Optional
import io
import torch
import time

from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector

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


    def _combine_key(
            self,
            key: Tuple[str, str],
        ) -> str:
        """
        Convert the tuple key to a single key
        """
        return ":".join(key)

    def _split_key(
            self,
            key: str,
        ) -> Tuple[str, str]:
        """
        Split the single key to a tuple key
        """
        return tuple(key.split(":"))

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
            key: Tuple[str, str]
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
            key: Tuple[str, str],
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
        with io.BytesIO() as f:
            start = time.perf_counter()
            torch.save(kv_chunk, f)
            start2 = time.perf_counter()
            f.seek(0)
            string = f.read()
            self.connection.set(self._combine_key(key), string)
            end = time.perf_counter()
            logger.info("Put %.2f MBytes data takes %.2f ms: serialization %.2fms, remote write %.2f ms", 
                        len(string) / 1e6, 
                        (end - start) * 1e3, 
                        (start2 - start) * 1e3, 
                        (end - start2) * 1e3)
        self.existing_keys.add(key)

    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[torch.Tensor]:
        """
        Retrive the KV cache chunk (in a single big tensor) by the given key
        """
        if not self.contains(key):
            return None

        with io.BytesIO() as f:
            start = time.perf_counter()
            data = self.connection.get(self._combine_key(key))
            end = time.perf_counter()
            if data is None or len(data) == 0:
                return None
            f.write(data)
            f.seek(0)
            ret = torch.load(f)
            end2 = time.perf_counter()
            logger.info("Get %.2f MBytes data takes %.2f ms: remote read %.2fms, deserialization %.2f ms", 
                        len(data) / 1e6, 
                        (end2 - start) * 1e3,
                        (end - start) * 1e3,
                        (end2 - end) * 1e3)
        return ret
