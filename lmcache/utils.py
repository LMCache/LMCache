import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
from nvtx import annotate  # type: ignore

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None


TORCH_DTYPE_TO_STR_DTYPE = {
    torch.half: "half",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double",
    torch.uint8: "fp8",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
}

@dataclass
class CacheManagerMetadata:
    context_id: List[str]
    method: str
    rate: float
    length: float # whole context's KV in bytes
    score_table: List[List[Tuple[float, float]]] # a list of score tables for each context, each table is a list of (rate, score) pairs

@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str
    metadata: CacheManagerMetadata

    def __hash__(self):
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
        ))
    
    def __eq__(self, other):
        if not isinstance(other, CacheEngineKey):
            return False
        return (
            self.fmt == other.fmt and
            self.model_name == other.model_name and
            self.world_size == other.world_size and
            self.worker_id == other.worker_id and
            self.chunk_hash == other.chunk_hash
        )

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]),
                              parts[4], CacheManagerMetadata([], "", 0.0, 0.0, []))


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


##### Threading related #####
def thread_safe(func):
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper
