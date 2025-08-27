# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
import asyncio
import hashlib
import threading
import traceback

# Third Party
from nvtx import annotate  # type: ignore
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: Optional[MemoryFormat] = None
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
    request_configs: Optional[dict] = None

    def __post_init__(self):
        tags = None
        if self.request_configs is not None:
            for k, v in self.request_configs.items():
                if k.startswith("lmcache.tag."):
                    if tags is None:
                        tags = {}
                    tags[k[len("lmcache.tag.") :]] = v
        self.tags = tags

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
                    self.request_configs,
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
            self.request_configs,
            0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 5:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 6:
            request_configs = {}
            for kv in parts[5:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs[kvs[0]] = kvs[1]
        return CacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4], 16),
            request_configs,
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
        if self.request_configs is not None and len(self.request_configs) != 0:
            msg["request_configs"] = [
                f"{k}%{v}" for k, v in self.request_configs.items()
            ]
        return msg

    @staticmethod
    def from_dict(d):
        request_configs = None
        if request_configs_list := d.get("request_configs"):
            request_configs = {}
            for kv in request_configs_list:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key dict: {d}")
                request_configs[kvs[0]] = kvs[1]
        return CacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            request_configs=request_configs,
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
                    self.request_configs,
                    layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 6:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 7:
            request_configs = {}
            for kv in parts[6:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs[kvs[0]] = kvs[1]
        return LayerCacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4], 16),
            request_configs,
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


#### Thread/asyncio-related utilities ####
def handle_thread_exception(args):
    logger.error(
        f"Thread {args.thread.name} crashed: {args.exc_type.__name__}: {args.exc_value}"
    )


def start_loop_in_thread_with_exceptions(loop: asyncio.AbstractEventLoop):
    # The loop must be set in the *same* thread where it runs.
    asyncio.set_event_loop(loop)

    # Catch unhandled exceptions from callbacks/tasks in this loop:
    def loop_excepthook(loop, context):
        msg = context.get("message", "Unhandled exception in event loop")
        exc = context.get("exception")
        logger.error(f"[asyncio] {msg}")
        if exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    loop.set_exception_handler(loop_excepthook)
    loop.run_forever()


#### Placeholder for dpsk broadcast functionality ####
def mock_up_broadcast_fn(t: torch.Tensor, i: int) -> None:
    raise NotImplementedError("Calling invalid broadcast function")


def mock_up_broadcast_object_fn(a: Any, i: int) -> None:
    raise NotImplementedError("Calling invalid broadcast object function")
