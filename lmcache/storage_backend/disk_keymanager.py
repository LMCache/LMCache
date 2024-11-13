import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

import yaml

from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCKeyManagerInterface
from lmcache.storage_backend.evictor import DummyEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import (CacheBackendInfo, LMCKeyManagerKey,
                           LMCKeyManagerValue)

logger = init_logger(__name__)


@dataclass
class LMCKeyManagerConfig:
    disk_url: Optional[str]
    disk_path: Optional[str]
    fmt: str
    dtype: str
    chunk_size: int
    serde: str

    @staticmethod
    def from_file(file_path: str) -> "LMCKeyManagerConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config = yaml.safe_load(fin)

        disk_url = config.get("disk_url", None)
        disk_path = config.get("disk_path", None)
        fmt = config.get("fmt", None)
        dtype = config.get("dtype", None)
        chunk_size = config.get("chunk_size", 256)
        serde = config.get("serde", None)

        return LMCKeyManagerConfig(disk_url, disk_path, fmt, dtype, chunk_size,
                                   serde)


class LMCDiskKeyManager(LMCKeyManagerInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(self, config: LMCKeyManagerConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__()
        self.info: CacheBackendInfo = CacheBackendInfo(config.fmt,
                                                       config.dtype,
                                                       config.chunk_size,
                                                       config.serde)

        self.dict: OrderedDict[LMCKeyManagerKey,
                               LMCKeyManagerValue] = OrderedDict()
        self.path = config.disk_path

        assert self.path is not None, ("Need to specify local path if when "
                                       "using  Local Disk")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.update_lock = threading.Lock()

        self.evictor = DummyEvictor()

    def contains(
        self,
        key_str: str,
    ) -> str:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        key = LMCKeyManagerKey.from_string(key_str)
        # return "Yes" if key in self.dict else "No"
        print("YES" if key in self.dict else "NO")
        # return key in self.dict
        return "YES" if key in self.dict else "NO"

    def _key_to_path(
        self,
        key: LMCKeyManagerKey,
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
        key: LMCKeyManagerKey,
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

    def put(self, key_str: str, kv_size: float, status: bool) -> str:
        print("PUT")
        print(key_str)
        key = LMCKeyManagerKey.from_string(key_str)
        if self.contains(key_str) == "YES" and status == 0:
            return ""
        if status == 1:
            self.dict[key].status = 2
            return ""

        path = self._key_to_path(key)

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, LMCKeyManagerValue(2, path, kv_size))

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return ""

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        self.update_lock.acquire()
        self.dict[key] = LMCKeyManagerValue(1, path, kv_size)
        self.update_lock.release()

        return path

    def get(
        self,
        key_str: str,
    ) -> LMCKeyManagerValue:
        #(TODO) Add read lock if needed
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        print("Point1")
        key = LMCKeyManagerKey.from_string(key_str)
        self.update_lock.acquire()
        if key not in self.dict:
            self.update_lock.release()
            print("Point2")
            return LMCKeyManagerValue(0, "", 0)

        if self.dict[key].status == 1:
            self.update_lock.release()
            print("Point3")
            return LMCKeyManagerValue(1, "", 0)

        self.evictor.update_on_get(key, self.dict)

        self.update_lock.release()
        print("Point4")
        return self.dict[key]

    def batched_get(
        self,
        keys_str: Iterable[str],
    ) -> Iterable[str]:
        #(TODO) Add read lock if needed
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        paths = []
        self.update_lock.acquire()
        for key_str in keys_str:
            key = LMCKeyManagerKey.from_string(key_str)
            if key not in self.dict:
                paths.append("")
                continue

            if self.dict[key].status == 1:
                paths.append("")
                continue

            self.evictor.update_on_get(key, self.dict)
            paths.append(self.dict[key].path)

        self.update_lock.release()
        return paths

    def Info(self):
        return self.info

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
        LMCDiskKeyManager(config))  # Register your class instance

    print("Server is running on port {}...".format(port))
    server.serve_forever()


if __name__ == '__main__':
    # Determine the directory of the current script (disk_keymanager.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to keymanager.yaml
    # based on disk_keymanager.py's location
    config_path = os.path.join(script_dir,
                               "../../examples/disk_backend/keymanager.yaml")
    print("Starting the server with the configuration file at", config_path)
    # Start the server with the configuration file
    start_server(LMCKeyManagerConfig.from_file(config_path))
