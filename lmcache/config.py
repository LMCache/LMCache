from dataclasses import dataclass
import re
from typing import Optional

@dataclass
class LMCacheEngineMetadata:
    model_name: str
    world_size: int
    worker_id: int

@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_device: str
    remote_url: str

    def from_defaults(
            chunk_size: int = 256,
            local_device: str = "cuda",
            remote_url: str = "redis://localhost:6379"
        ) -> 'LMCacheEngineConfig':
        return LMCacheEngineConfig(chunk_size, local_device, remote_url)

    def from_legacy(
            chunk_size: int = 256,
            backend: str = "cuda",
            persist_path: str = None
        ) -> 'LMCacheEngineConfig':
        match backend:
            case "cpu" | "cuda":
                local_device = backend
                remote_url = None
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                local_device = None
                remote_url = url 
        return LMCacheEngineConfig(chunk_size, local_device, remote_url)

### SOME GLOBAL CONFIGS 
# TODO: it needs to be manually updated in the code here, but cannot be really configured
class GlobalConfig:
    enable_debug: bool = True

    @classmethod
    def set_debug(cls, enable: bool):
        cls.enable_debug = enable

    @classmethod
    def is_debug(cls) -> bool:
        return cls.enable_debug
