# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, OrderedDict, Tuple
import hashlib
import threading

# Third Party
from nvtx import annotate  # type: ignore
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: MemoryFormat = None
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check if the disk cache can be evicted.
        """
        return not self.is_pinned


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

STR_DTYPE_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_STR_DTYPE.items()}


@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    tags: Optional[OrderedDict] = None

    def __hash__(self):
        if self.tags is None:
            return hash(
                (
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                )
            )
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                "%".join([f"{k}={v}" for k, v in self.tags.items()]),
            )
        )

    def to_string(self):
        s = (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags.items()]
            s += "@" + "@".join(tags)
        return s

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    self.tags,
                    layer_id,
                )
            )
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """Return the key for the first layer"""
        key = LayerCacheEngineKey(
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            self.tags,
            0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 5:
            raise ValueError(f"Invalid key string: {s}")
        tags = None
        if len(parts) >= 6:
            tags = OrderedDict()
            for kv in parts[5:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                tags[kvs[0]] = kvs[1]
        return CacheEngineKey(
            parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), tags
        )

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        msg = {
            "__type__": "CacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
        }
        if self.tags is not None and len(self.tags) != 0:
            msg["tags"] = [f"{k}%{v}" for k, v in self.tags.items()]
        return msg

    @staticmethod
    def from_dict(d):
        tags = None
        if tag_list := d.get("tags"):
            tags = OrderedDict()
            for kv in tag_list:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key dict: {d}")
                tags[kvs[0]] = kvs[1]
        return CacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            tags=tags,
        )


@dataclass(order=True)
class LayerCacheEngineKey(CacheEngineKey):
    """A key for the layer cache engine"""

    layer_id: int = 0

    def __hash__(self):
        if self.tags is None:
            return hash(
                (
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    self.layer_id,
                )
            )
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                "%".join([f"{k}={v}" for k, v in self.tags.items()]),
                self.layer_id,
            )
        )

    def to_string(self):
        s = (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags.items()]
            s += "@" + "@".join(tags)
        return s

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    self.tags,
                    layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 6:
            raise ValueError(f"Invalid key string: {s}")
        tags = None
        if len(parts) >= 7:
            tags = OrderedDict()
            for kv in parts[6:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                tags[kvs[0]] = kvs[1]
        return LayerCacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            tags,
            int(parts[5]),
        )


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


##### Observability Threading related #####
_shared_observability_lock = threading.Lock()


def thread_safe(func):
    def wrapper(*args, **kwargs):
        with _shared_observability_lock:
            result = func(*args, **kwargs)
        return result

    return wrapper
