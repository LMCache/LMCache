import os
import re
import threading
from collections import OrderedDict
from socketserver import ThreadingMixIn
from typing import Iterable
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from lmcache.config import LMCAddressManagerConfig
from lmcache.logging import init_logger
from lmcache.storage_backend.evictor import LRUEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import CacheEngineKey, DiskCacheMetadata

logger = init_logger(__name__)


class LMCDiskAddressManager():
    """
    The address managers centralize the key-path dictionary and 
        the evictor of each disk engine into an independent process, 
        enabling multiple LMCache Disk Backends to share KV Cache files.
    """

    def __init__(self, config: LMCAddressManagerConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """

        # Dict key to path & size
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()

        self.path = config.disk_path

        assert self.path is not None, ("Need to specify local path if when "
                                       "using  Local Disk")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.update_lock = threading.Lock()

        self.evictor = LRUEvictor()

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.to_string().replace("/", "-") + ".pt"

    def remove(
        self,
        key: CacheEngineKey,
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

    def contains(
        self,
        key_str: str,
    ) -> str:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            path if the cache engine contains the key, "" otherwise
        """
        key = CacheEngineKey.from_string(key_str)
        with self.update_lock:
            return self.dict[key].path if key in self.dict else ""

    def batched_contains(
        self,
        keys_strs: Iterable[str],
    ) -> Iterable[str]:
        return [self.contains(key_str) for key_str in keys_strs]

    def write_check(self, key_str: str, kv_size: float) -> str:
        """
        Principal: Only the backend that first encounter the key 
            should write it to the disk

        return "" if the key doesn't need to be write again,
            when the key already written or 
            other backend is writing the same KV Cache chunks,
            Backend SHOULD NOT write it again
        return path if the key needs to be write, 
            when the key is new and not in the dict
            Backend SHOULD write it to the path in the later put function

        kv_size: the size of the KV Cache chunk group in GB, 
            free up the space if the existing cache is too big

        if path is not in the cache, initialize the cache with ("",0), so that 
            when other backend try to write the same KV Cache chunks, 
                it will be blocked by the "" value
            when other backend try to read the same KV Cache chunks, 
                it will be blocked by the "" value
        """
        key = CacheEngineKey.from_string(key_str)
        with self.update_lock:
            if key in self.dict:
                return ""

            evict_keys, put_status = self.evictor.update_on_put(
                self.dict, int(kv_size * (1024 * 1024)))

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return ""

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        with self.update_lock:
            self.dict[key] = DiskCacheMetadata("", 0)

        return self._key_to_path(key)

    def batched_write_check(self, key_strs: Iterable[str],
                            kv_size: float) -> Iterable[str]:
        paths = []
        kv_size = int(kv_size * (1024 * 1024))

        self.update_lock.acquire()
        for key_str in key_strs:
            key = CacheEngineKey.from_string(key_str)
            if key in self.dict:
                # Do not free up space for existed KV Cache chunks
                kv_size = kv_size - self.dict[key].size
                paths.append("")
            else:
                self.dict[key] = DiskCacheMetadata("", 0)
                paths.append(self._key_to_path(key))

        evict_keys, put_status = self.evictor.update_on_put(self.dict, kv_size)

        self.update_lock.release()

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            # print("Illegal")
            return [""] * len(key_strs)  # type: ignore

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        return paths

    def write_ready(self, key_str: str, kv_size: float):
        """
        When backend has finished writing the cache, 
            it will call this function to update the cache 
            from ("",0) to the actual path and size
        
        kv_size: the size of the KV Cache chunk in GB,
            for evictor
        """
        key = CacheEngineKey.from_string(key_str)

        with self.update_lock:
            self.dict[key] = DiskCacheMetadata(self._key_to_path(key),
                                               int(kv_size * (1024 * 1024)))

        return True

    def batched_write_ready(self, key_strs: Iterable[str],
                            kv_sizes: Iterable[float]):
        self.update_lock.acquire()
        for key_str, kv_size in zip(key_strs, kv_sizes):
            key = CacheEngineKey.from_string(key_str)
            self.dict[key] = DiskCacheMetadata(self._key_to_path(key),
                                               int(kv_size * (1024 * 1024)))
        self.update_lock.release()
        return True

    def read_check(self, key_str: str) -> str:
        """
        return the path if the key is in the cache
            Backend SHOULD read the cache from the path
        
        return "" otherwise, including cache that is 
            being written by other backend
            Backend SHOULD NOT read this cache
        """
        self.update_lock.acquire()
        key = CacheEngineKey.from_string(key_str)
        if key not in self.dict:
            self.update_lock.release()
            return ""

        self.evictor.update_on_get(key, self.dict)

        path = self.dict[key].path
        self.update_lock.release()
        return path

    def batched_read_check(self, key_strs: Iterable[str]) -> Iterable[str]:
        paths = []
        self.update_lock.acquire()
        for key_str in key_strs:
            key = CacheEngineKey.from_string(key_str)
            if key not in self.dict:
                paths.append("")
                continue

            self.evictor.update_on_get(key, self.dict)
            paths.append(self.dict[key].path)
        self.update_lock.release()
        return paths

    def clear(self):
        self.update_lock.acquire()
        # Remove all files
        for key in self.dict:
            os.remove(self.dict[key].path)
        self.dict.clear()
        self.update_lock.release()
        return True


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', )


class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


# Initialize the server
def start_server(config):
    pattern = r"(.+)://(.*):(.*)"
    m = re.match(pattern, config.disk_url)
    if m is None:
        logger.error(f"Cannot parse disk url {config.disk_url} in the config")
        raise ValueError(f"Invalid disk url {config.disk_url}")

    connector_type, host, port = m.group(1), m.group(2), int(m.group(3))
    print(connector_type, host, port)
    # server = SimpleXMLRPCServer((host, port), requestHandler=RequestHandler)
    server = ThreadedXMLRPCServer((host, port), requestHandler=RequestHandler)
    server.register_introspection_functions(
    )  # Optional: provides a list of registered functions
    server.register_instance(
        LMCDiskAddressManager(config))  # Register your class instance

    print("Server is running on port {}...".format(port))
    server.serve_forever()


def main():
    if "LMCACHE_CONFIG_FILE" not in os.environ:
        config = LMCAddressManagerConfig("disk_url://localhost:4322",
                                         "/local_disk/")
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        config = LMCAddressManagerConfig.from_file(config_file)
    start_server(config)


if __name__ == "__main__":
    main()
