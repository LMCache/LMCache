# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import asyncio
import hashlib
import re
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


def compress_slot_mapping(slots: list[int]) -> list[Union[int, list[int]]]:
    """Compress a list of slot indices into ranges while preserving order.

    Consecutive slots (3 or more) are represented as [start, end] ranges.
    Single elements or pairs are kept as individual integers.
    For example: [1, 2, 3, 4, 5, 9, 10, 11, 12] -> [[1, 5], [9, 12]]
    Order-preserving: [5, 3, 1, 2, 4] -> [5, 3, 1, 2, 4] (no compression)
    Mixed: [1, 2, 3, 4, 5, 7, 8] -> [[1, 5], 7, 8]

    Args:
        slots: List of slot indices (order is preserved).

    Returns:
        List of integers or [start, end] ranges. Ranges are only used
        when there are 3 or more consecutive elements.
    """
    if not slots:
        return []

    result: list[Union[int, list[int]]] = []
    range_start = slots[0]
    range_end = slots[0]

    for slot in slots[1:]:
        if slot == range_end + 1:
            # Extend current range
            range_end = slot
        else:
            # Close current range and start a new one
            _append_range_or_elements(result, range_start, range_end)
            range_start = slot
            range_end = slot

    # Append the last range
    _append_range_or_elements(result, range_start, range_end)
    return result


def _append_range_or_elements(
    result: list[Union[int, list[int]]], start: int, end: int
) -> None:
    """Helper to append range or individual elements based on length.

    Only compresses to [start, end] if there are 3 or more consecutive elements.
    """
    length = end - start + 1
    if length >= 3:
        # Compress: 3 or more consecutive elements
        result.append([start, end])
    else:
        # Don't compress: 1 or 2 elements
        for i in range(start, end + 1):
            result.append(i)


def decompress_slot_mapping(compressed: list[Union[int, list[int]]]) -> list[int]:
    """Decompress slot ranges back to a list of slot indices.

    Inverse operation of compress_slot_mapping.
    For example: [[1, 5], [9, 12]] -> [1, 2, 3, 4, 5, 9, 10, 11, 12]
    Mixed: [[1, 5], 7, 8] -> [1, 2, 3, 4, 5, 7, 8]

    Args:
        compressed: List of integers or [start, end] ranges from
            compress_slot_mapping.

    Returns:
        List of slot indices.
    """
    slots: list[int] = []
    for item in compressed:
        if isinstance(item, list):
            start, end = item
            slots.extend(range(start, end + 1))
        else:
            slots.append(item)
    return slots


def parse_mixed_slot_mapping(
    slot_mapping_str: str,
) -> Tuple[Optional[list[int]], Optional[dict]]:
    """Parse mixed format slot_mapping string.

    Supports two formats:
    1. Single numbers: "1,2,3,17,19"
    2. Range format: "[9,12]" (represents 9,10,11,12)
    3. Mixed format: "1,2,3,[9,12],17,19" (represents 1,2,3,9,10,11,12,17,19)

    Args:
        slot_mapping_str: String containing slot mapping information.

    Returns:
        Tuple of (slot_indices list, error dict).
        If error dict is not None, slot_indices will be None.
    """
    try:
        # Remove all whitespace
        clean_str = "".join(slot_mapping_str.split())

        # Split by comma but preserve range expressions
        parts = []
        buffer = ""
        in_brackets = False

        for char in clean_str:
            if char == "[":
                if in_brackets:
                    raise ValueError("Nested brackets not allowed")
                in_brackets = True
                buffer += char
            elif char == "]":
                if not in_brackets:
                    raise ValueError("Unmatched closing bracket")
                in_brackets = False
                buffer += char
                parts.append(buffer)
                buffer = ""
            elif char == "," and not in_brackets:
                if buffer:
                    parts.append(buffer)
                    buffer = ""
            else:
                buffer += char

        # Add the last part if any
        if buffer:
            parts.append(buffer)

        if in_brackets:
            raise ValueError("Unclosed bracket")

        # Parse each part
        compressed: list[Union[int, list[int]]] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if it's a range format [start,end]
            range_match = re.match(r"^\[(\d+),(\d+)\]$", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start > end:
                    raise ValueError(f"Range start {start} must be <= end {end}")
                compressed.append([start, end])
            else:
                # Single number
                try:
                    num = int(part)
                    compressed.append(num)
                except ValueError as ve:
                    raise ValueError(f"Invalid slot format: '{part}'") from ve

        # Decompress to individual slot indices
        slot_indices = decompress_slot_mapping(compressed)
        return slot_indices, None

    except Exception as e:
        return None, {
            "error": "Invalid slot_mapping format",
            "message": (
                f"slot_mapping must be comma-separated integers "
                f"or ranges like [start,end]: {str(e)}"
            ),
        }


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
