# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import asyncio
import hashlib
import threading
import traceback

try:
    # Third Party
    from nvtx import annotate  # type: ignore
except ImportError:

    def annotate(*args, **kwargs):
        """Dummy decorator when nvtx is not available."""

        def decorator(func):
            return func

        return decorator


# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


# Math utility functions
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def round_down(x: int, y: int) -> int:
    """Round down x to the nearest multiple of y."""
    return (x // y) * y


try:
    # First Party
    from lmcache import _version  # type: ignore[attr-defined]

    VERSION = getattr(_version, "__version__", "")
    COMMIT_ID = getattr(_version, "__commit_id__", "")
except ImportError:
    VERSION = ""
    COMMIT_ID = ""


def get_version():
    version_display = VERSION if VERSION else "NA"
    commit_id_display = COMMIT_ID if COMMIT_ID else "NA"
    return f"{version_display}-{commit_id_display}"


def convert_tokens_to_list(
    tokens: Optional[Union[torch.Tensor, list[int]]], token_start: int, token_end: int
) -> List[int]:
    """Convert tokens to a list.
    token_start and token_end delineate tokens to convert"""
    if tokens is None:
        return []

    return (
        tokens.tolist()[token_start : token_end + 1]
        if isinstance(tokens, torch.Tensor)
        else tokens[token_start : token_end + 1]
    )


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    cached_positions: Optional[torch.Tensor] = None
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
    torch.double: "double",
    torch.float64: "double",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}

# FP8 variants (PyTorch ≥2.1)
if hasattr(torch, "float8_e4m3fn"):
    TORCH_DTYPE_TO_STR_DTYPE[torch.float8_e4m3fn] = "fp8_e4m3"
if hasattr(torch, "float8_e4m3fnuz"):
    TORCH_DTYPE_TO_STR_DTYPE[torch.float8_e4m3fnuz] = "fp8_e4m3"
if hasattr(torch, "float8_e5m2"):
    TORCH_DTYPE_TO_STR_DTYPE[torch.float8_e5m2] = "fp8_e5m2"
if hasattr(torch, "float8_e5m2fnuz"):
    TORCH_DTYPE_TO_STR_DTYPE[torch.float8_e5m2fnuz] = "fp8_e5m2"

STR_DTYPE_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_STR_DTYPE.items()}


def parse_cache_key(key_str: str) -> Union[CacheEngineKey, LayerCacheEngineKey]:
    """Parse a key string into either a CacheEngineKey or LayerCacheEngineKey.

    Args:
        key_str: String in format:
            fmt@model@world_size@worker_id@chunk_hash[@layer_id][@tag%value...]

    Returns:
        CacheEngineKey if no layer_id, LayerCacheEngineKey if valid layer_id
    """
    parts = key_str.strip().split("@")
    if len(parts) >= 6 and parts[5].isdigit():
        return LayerCacheEngineKey.from_string(key_str)
    return CacheEngineKey.from_string(key_str)


@dataclass(slots=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = field(default_factory=dict)
    tags: Optional[tuple] = field(init=False, default=None)
    _dtype_str: str = field(init=False, default="")

    def __post_init__(self):
        tag_list = None
        if self.request_configs is not None:
            for k, v in self.request_configs.items():
                if k.startswith("lmcache.tag."):
                    if tag_list is None:
                        tag_list = []
                    tag_list.append((k[len("lmcache.tag.") :], v))
        if self.dtype not in TORCH_DTYPE_TO_STR_DTYPE:
            raise ValueError(f"Unsupported dtype in CacheEngineKey: {self.dtype}")
        self._dtype_str = TORCH_DTYPE_TO_STR_DTYPE[self.dtype]
        # use tuple to save tags
        self.tags = None if tag_list is None else tuple(tag_list)

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self._dtype_str,
                self.tags,
            )
        )

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.fmt == other.fmt
                and self.model_name == other.model_name
                and self.world_size == other.world_size
                and self.worker_id == other.worker_id
                and self.chunk_hash == other.chunk_hash
                and self.dtype == other.dtype
                and self.tags == other.tags
            )

        return False

    def to_string(self):
        s = (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash:x}@{self._dtype_str}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags]
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
                    self.dtype,
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
            self.dtype,
            self.request_configs,
            0,
        )
        return key

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
                request_configs["lmcache.tag." + kvs[0]] = kvs[1]
        return CacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4], 16),
            STR_DTYPE_TO_TORCH_DTYPE[parts[5]],
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
            "dtype": self._dtype_str,
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
            dtype=STR_DTYPE_TO_TORCH_DTYPE[d["dtype"]],
            request_configs=request_configs,
        )

    def with_new_worker_id(self, new_worker_id: int) -> "CacheEngineKey":
        # Reconstruct the cache engine key with new worker id
        return CacheEngineKey(
            self.fmt,
            self.model_name,
            self.world_size,
            new_worker_id,
            self.chunk_hash,
            self.dtype,
            self.request_configs,
        )


@dataclass(slots=True)
class LayerCacheEngineKey(CacheEngineKey):
    """A key for the layer cache engine"""

    layer_id: int = 0

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self._dtype_str,
                self.tags,
                self.layer_id,
            )
        )

    def __eq__(self, other):
        if super(LayerCacheEngineKey, self).__eq__(other):
            return self.layer_id == other.layer_id

        return False

    def to_string(self):
        s = (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash:x}@{self._dtype_str}@{self.layer_id}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags]
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
                    self.dtype,
                    self.request_configs,
                    layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 7:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 8:
            request_configs = {}
            for kv in parts[7:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs["lmcache.tag." + kvs[0]] = kvs[1]
        return LayerCacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4], 16),
            STR_DTYPE_TO_TORCH_DTYPE[parts[5]],
            request_configs,
            int(parts[6]),
        )


@dataclass
class CacheStoreEvent:
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None


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
