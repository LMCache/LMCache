import torch
import hashlib
from nvtx import annotate
from typing import Tuple
from dataclasses import dataclass

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

@dataclass
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash((self.fmt, self.model_name, self.world_size, self.worker_id, self.chunk_hash))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4])


##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]

def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]

def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in cudf."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)
