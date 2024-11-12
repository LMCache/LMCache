import hashlib
from dataclasses import dataclass
from typing import Tuple

import torch
from nvtx import annotate  # type: ignore

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "double": torch.float64,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

@dataclass
class DiskCacheMetadata:
    path: str
    size: float


@dataclass
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
        ))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]),
                              parts[4])

@dataclass
class CacheBackendInfo:
    fmt:str
    dtype:str
    chunk_size:int
    serde:str

@dataclass
class LMCKeyManagerKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str
    def __hash__(self):
        return hash((
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
        ))
    def to_string(self):
        return f"{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s:str):
        parts = s.split("@")
        if len(parts) != 4:
            raise ValueError(f"Invalid key string: {s}")
        return LMCKeyManagerKey(parts[0], int(parts[1]), int(parts[2]),
                              parts[3])

@dataclass
class LMCKeyManagerValue:
    """
        status:0~3
            0: not exist
            1: writing
            2: exist and ready to read
            3: reading
        path: path/url
    """
    status:int
    path:str
    size: float

##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)