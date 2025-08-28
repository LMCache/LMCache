# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import List, Optional
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.server.server_storage_backend.abstract_backend import (
    LMSBackendInterface,
)
from lmcache.storage_backend.evictor import DummyEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import DiskCacheMetadata, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class LMSLocalBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory.
    """

    def __init__(
        self,
    ) -> None:
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """
        super().__init__()

        self.dict: OrderedDict[str, bytearray] = OrderedDict()

        self.update_lock = threading.Lock()

        self.evictor = DummyEvictor()

    def list_keys(self) -> List[str]:
        return list(self.dict.keys())

    def contains(
        self,
        key: str,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def remove(
        self,
        key: str,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """
        self.dict.pop(key)

    def put(
        self,
        key: str,
        kv_chunk_bytes: bytearray,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk_bytes: the kv cache of the token chunk, in the format of
            bytearray

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")

        self.update_lock.acquire()
        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.evictor.get_size(kv_chunk_bytes)
        )

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            self.update_lock.release()
            return

        # Evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        # Store new chunk
        self.dict[key] = kv_chunk_bytes
        self.update_lock.release()

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: str,
    ) -> Optional[bytearray]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        self.update_lock.acquire()
        kv_chunk = self.dict.get(key, None)

        # Update cache recency
        if kv_chunk is not None:
            self.evictor.update_on_get(key, self.dict)

        self.update_lock.release()
        return kv_chunk

    def close(self):
        pass


# TODO(Jiayi): need to optimize disk loading
# current impl. with "naive open read/write" might not be efficient
# (better than torch.load)
class LMSLocalDiskBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(
        self,
        path: str,
    ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """
        super().__init__()

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.dict: OrderedDict[str, DiskCacheMetadata] = OrderedDict()

        self.update_lock = threading.Lock()

        self.evictor = DummyEvictor()

    def list_keys(self) -> List[str]:
        return list(self.dict.keys())

    def contains(
        self,
        key: str,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def _key_to_path(
        self,
        key: str,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.replace("/", "-") + ".bin"

    def remove(
        self,
        key: str,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """
        self.update_lock.acquire()
        path = self.dict[key].path
        self.dict.pop(key)
        self.update_lock.release()

        os.remove(path)

    def put(
        self,
        key: str,
        kv_chunk_bytes: bytearray,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        path = self._key_to_path(key)

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, self.evictor.get_size(kv_chunk_bytes)
        )

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        logger.info(f"Saving cache to {path}")
        # torch.save(kv_chunk_bytes, self._key_to_path(key))
        with open(self._key_to_path(key), "wb") as binary_file:
            binary_file.write(kv_chunk_bytes)

        self.update_lock.acquire()
        self.dict[key] = DiskCacheMetadata(path, self.evictor.get_size(kv_chunk_bytes))
        self.update_lock.release()

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: str,
    ) -> Optional[bytes]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        self.update_lock.acquire()
        if key not in self.dict:
            self.update_lock.release()
            return None

        path = self.dict[key].path
        self.evictor.update_on_get(key, self.dict)

        with open(path, "rb") as binary_file:
            kv_chunk = binary_file.read()
        self.update_lock.release()
        return kv_chunk

        # return torch.load(self._key_to_path(key))

    def close(self):
        pass
