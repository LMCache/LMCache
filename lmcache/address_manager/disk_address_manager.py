import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

import yaml

from lmcache.logging import init_logger
from lmcache.storage_backend.evictor import LRUEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import CacheEngineKey, DiskCacheMetadata

logger = init_logger(__name__)


@dataclass
class LMCAddressManagerConfig:
    disk_url: Optional[str]
    disk_path: Optional[str]

    @staticmethod
    def from_file(file_path: str) -> "LMCAddressManagerConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config = yaml.safe_load(fin)

        local_device = config.get("local_device",
                                  "disk_url://http://localhost:4322")
        disk_url = local_device[11:]
        disk_path = config.get("disk_path", "/local/local_disk/")

        return LMCAddressManagerConfig(disk_url, disk_path)


class LMCDiskAddressManager():
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(self, config: LMCAddressManagerConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__()

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
        return self.dict[key].path if key in self.dict else ""

    def batched_contains(
        self,
        keys_strs: Iterable[str],
    ) -> Iterable[str]:
        return [self.contains(key_str) for key_str in keys_strs]

    def write_check(self, key_str: str, kv_size: float) -> str:
        """
        The logic here is that:
            return "" if the key is already in the cache, 
                so that backend will not write it again
            return path if the key is not in the cache, 
                so that backend will write it to the path
        if path is not in the cache, initialize the cache with ("",0)
        """
        key = CacheEngineKey.from_string(key_str)
        if key in self.dict:
            return ""
        
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, DiskCacheMetadata("", kv_size))

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return ""

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        self.dict[key] = DiskCacheMetadata("", 0)
        return self._key_to_path(key)

    def batched_write_check(self, key_strs: Iterable[str],
                            kv_size: float) -> Iterable[str]:
        paths = []
        for key_str in key_strs:
            key = CacheEngineKey.from_string(key_str)
            if key in self.dict:
                kv_size = kv_size - self.dict[key].size
                paths.append("")
            else:
                self.dict[key] = DiskCacheMetadata("", 0)
                paths.append(self._key_to_path(key))

        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, DiskCacheMetadata("", kv_size))

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
        """
        key = CacheEngineKey.from_string(key_str)
        self.dict[key] = DiskCacheMetadata(self._key_to_path(key), kv_size)
        return True

    def batched_write_ready(self, key_strs: Iterable[str],
                            kv_sizes: Iterable[float]):
        for key_str, kv_size in zip(key_strs, kv_sizes):
            key = CacheEngineKey.from_string(key_str)
            self.dict[key] = DiskCacheMetadata(self._key_to_path(key), kv_size)
        return True

    def read_check(self, key_str: str) -> str:
        """
        return the path if the key is in the cache, 
            "" otherwise, including cache that is not ready
        """
        self.update_lock.acquire()
        key = CacheEngineKey.from_string(key_str)
        if key not in self.dict:
            self.update_lock.release()
            return ""

        self.evictor.update_on_get(key, self.dict)

        self.update_lock.release()
        return self.dict[key].path

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

    # def Info(self):
    #     return self.info

    def close(self):
        pass

    def __del__(self):
        self.close()


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', )


# Initialize the server
def start_server(config):
    pattern = r"(.+)://(.*):(.*)"
    m = re.match(pattern, config.disk_url)
    if m is None:
        logger.error(f"Cannot parse disk url {config.disk_url} in the config")
        raise ValueError(f"Invalid disk url {config.disk_url}")

    connector_type, host, port = m.group(1), m.group(2), int(m.group(3))
    print(connector_type, host, port)
    server = SimpleXMLRPCServer((host, port), requestHandler=RequestHandler)
    server.register_introspection_functions(
    )  # Optional: provides a list of registered functions
    server.register_instance(
        LMCDiskAddressManager(config))  # Register your class instance

    print("Server is running on port {}...".format(port))
    server.serve_forever()


def main():
    if "LMCACHE_CONFIG_FILE" not in os.environ:
        config = LMCAddressManagerConfig("http://localhost:4322",
                                         "/local/local_disk/")
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        config = LMCAddressManagerConfig.from_file(config_file)
    start_server(config)


if __name__ == "__main__":
    main()
