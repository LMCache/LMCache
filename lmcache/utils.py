import hashlib
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    is_pin: bool = False

    def pin(self) -> bool:
        self.is_pin = True
        return True

    def unpin(self) -> bool:
        self.is_pin = False
        return True

    @property
    def is_pinned(self) -> bool:
        return self.is_pin


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


@dataclass(order=True)
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

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """ Split the key into multiple keys for each layer """
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(self.fmt, self.model_name, self.world_size,
                                    self.worker_id, self.chunk_hash, layer_id))
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """ Return the key for the first layer """
        key = LayerCacheEngineKey(self.fmt, self.model_name, self.world_size,
                                  self.worker_id, self.chunk_hash, 0)
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]),
                              parts[4])


@dataclass(order=True)
class LayerCacheEngineKey(CacheEngineKey):
    """ A key for the layer cache engine """
    layer_id: int

    def __hash__(self):
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            self.layer_id,
        ))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 6:
            raise ValueError(f"Invalid key string: {s}")
        return LayerCacheEngineKey(parts[0], parts[1], int(parts[2]),
                                   int(parts[3]), parts[4], int(parts[5]))


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
