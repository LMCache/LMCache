# SPDX-License-Identifier: Apache-2.0
"""``remote_serde: fp8lin`` — lossless predictive coding of FP8 KV chunks.

Stores each KV chunk as E4M3 codes entropy coded against a fitted linear
prediction (other heads of the same token, previous layer of the same token), on
the GPU. Typical size is 4.2-4.8 bits per value, i.e. 1.7-1.9x smaller than FP8
and 3.4-3.8x smaller than bf16.

* FP8 KV caches (vLLM ``kv_cache_dtype="fp8"``; chunks arrive as ``uint8`` or
  ``float8_e4m3fn``) are stored **losslessly**: the decoded bytes equal the input.
* bf16 / fp16 KV caches are first rounded to E4M3 with the coder's per-layer scales
  (lossy, like the ``fp8`` serde), then coded losslessly.

Configuration (``extra_config``)::

    fp8lin.coder_path: /path/to/coder   # required; see fp8lin/coder.py
    fp8lin.device: cuda:0               # optional; default: current CUDA device

A coder directory is fitted once per model (and KV dtype convention) on a few
hundred documents; see ``lmcache/v1/fp8lin/README.md``.
"""

# Standard
from pathlib import Path
from typing import Optional
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.fp8lin import ChunkCodec
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer

logger = init_logger(__name__)

FP8_STORAGE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
E4M3_MAX = 448.0

_CODECS: dict[tuple[str, str], tuple[ChunkCodec, threading.Lock]] = {}
_CODECS_LOCK = threading.Lock()


def _shared_codec(config: LMCacheEngineConfig) -> tuple[ChunkCodec, threading.Lock]:
    """One codec (and lock) per coder directory and device, shared by the
    serializer and the deserializer of a process.

    Raises:
        ValueError: ``fp8lin.coder_path`` is missing from ``extra_config``.
    """
    extra = config.extra_config or {}
    path = extra.get("fp8lin.coder_path")
    if not path:
        raise ValueError(
            "remote_serde=fp8lin requires extra_config['fp8lin.coder_path']"
        )
    dev = extra.get("fp8lin.device")
    device = str(dev) if dev else f"{torch_device_type}:{torch_dev.current_device()}"
    key = (str(Path(path).resolve()), device)
    with _CODECS_LOCK:
        if key not in _CODECS:
            _CODECS[key] = (ChunkCodec(path, device), threading.Lock())
        return _CODECS[key]


def _scales(codec: ChunkCodec) -> torch.Tensor:
    """Per-layer E4M3 scales [2, L, 1, 1] (K, V) used for non-FP8 KV caches."""
    s = codec.meta.get("kv_scale") or {}
    k = s.get("k") or [1.0] * codec.L
    v = s.get("v") or [1.0] * codec.L
    return torch.tensor([k, v], dtype=torch.float32, device=codec.device)[
        ..., None, None
    ]


class FP8LinSerializer(Serializer):
    """Encode KV chunks ``[2, num_layers, num_tokens, hidden]`` into fp8lin blobs.

    Args:
        config: LMCache engine config (reads ``extra_config`` and ``chunk_size``).
        metadata: LMCache engine metadata (KV dtype).
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        self.codec, self.lock = _shared_codec(config)
        self.kv_dtype = metadata.kv_dtype
        self.align = int(config.chunk_size)
        self.scale = _scales(self.codec)

    def _codes(self, memory_obj: MemoryObj) -> torch.Tensor:
        """uint8 E4M3 codes [L, T, 2C] on the codec device."""
        t = memory_obj.tensor
        if t is None:
            raise ValueError("fp8lin: memory object has no tensor")
        if t.dim() != 4 or t.shape[0] != 2 or t.shape[1] != self.codec.L:
            raise ValueError(
                f"fp8lin: expected [2, {self.codec.L}, T, C], got {t.shape}"
            )
        if t.shape[3] != self.codec.C:
            raise ValueError(
                f"fp8lin: hidden size {t.shape[3]} != coder {self.codec.C}"
            )
        t = t.to(self.codec.device, non_blocking=True)
        if t.dtype in FP8_STORAGE_DTYPES:
            c = t.view(torch.uint8)
        else:
            x = (t.float() / self.scale).clamp(-E4M3_MAX, E4M3_MAX)
            c = x.to(torch.float8_e4m3fn).view(torch.uint8)
        L, T, C = t.shape[1], t.shape[2], t.shape[3]
        return c.permute(1, 2, 0, 3).reshape(L, T, 2 * C).contiguous()

    @staticmethod
    def _start(memory_obj: MemoryObj, T: int) -> Optional[int]:
        """Chunk start from ``cached_positions`` when they are present and
        consecutive; else ``None`` (recovered from the keys)."""
        cp = memory_obj.metadata.cached_positions
        if cp is None or cp.numel() != T:
            return None
        first, last = int(cp[0]), int(cp[-1])
        return first if last - first == T - 1 else None

    def batched_serialize(self, memory_objs: list[MemoryObj]) -> list[MemoryObj]:
        """Encode several chunks in one GPU pass.

        Args:
            memory_objs: KV chunks in ``MemoryFormat.KV_2LTD``.

        Returns:
            One ``BytesBufferMemoryObj`` per chunk, in order.
        """
        with self.lock:
            codes = [self._codes(m) for m in memory_objs]
            starts = [
                self._start(m, c.shape[1])
                for m, c in zip(memory_objs, codes, strict=True)
            ]
            missing = [i for i, s in enumerate(starts) if s is None]
            if missing:
                found = self.codec.find_starts([codes[i] for i in missing], self.align)
                for i, s in zip(missing, found, strict=True):
                    starts[i] = s
            blobs = self.codec.encode(codes, [int(s) for s in starts])  # type: ignore[arg-type]
        return [BytesBufferMemoryObj(b) for b in blobs]

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        """Encode one KV chunk into an fp8lin blob (see :meth:`batched_serialize`)."""
        return self.batched_serialize([memory_obj])[0]


class FP8LinDeserializer(Deserializer):
    """Decode fp8lin blobs back into KV chunks in the engine's KV dtype.

    Args:
        config: LMCache engine config (reads ``extra_config``).
        metadata: LMCache engine metadata (KV dtype).
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        self.codec, self.lock = _shared_codec(config)
        self.kv_dtype = metadata.kv_dtype
        self.scale = _scales(self.codec)

    def _to_kv(self, codes: torch.Tensor) -> MemoryObj:
        L, T, C2 = codes.shape
        kv = codes.view(L, T, 2, C2 // 2).permute(2, 0, 1, 3).contiguous()
        if self.kv_dtype in FP8_STORAGE_DTYPES:
            kv = kv.view(self.kv_dtype)
        else:
            kv = (kv.view(torch.float8_e4m3fn).float() * self.scale).to(self.kv_dtype)
        return TensorMemoryObj(
            raw_data=kv,
            metadata=MemoryObjMetadata(
                shape=kv.shape,
                dtype=kv.dtype,
                address=-1,
                phy_size=kv.numel() * kv.element_size(),
                ref_count=-1,  # not owned by an allocator (as for CacheGen)
                fmt=MemoryFormat.KV_2LTD,
            ),
            parent_allocator=None,
        )

    def batched_deserialize(
        self, memory_objs: list[MemoryObj]
    ) -> list[Optional[MemoryObj]]:
        """Decode several blobs in one GPU pass.

        Returns:
            One KV chunk per blob, or ``None`` where the decoded codes fail the
            blob's checksum (treated as a cache miss).
        """
        blobs = [bytes(m.byte_array) for m in memory_objs]
        with self.lock:
            try:
                codes = self.codec.decode(blobs)
            except ValueError:  # a malformed blob in the batch: decode one by one
                codes = []
                for b in blobs:
                    try:
                        codes.append(self.codec.decode([b])[0])
                    except ValueError:
                        logger.warning("fp8lin: malformed blob, dropping a chunk")
                        codes.append(None)
        out: list[Optional[MemoryObj]] = []
        for c in codes:
            if c is None:
                logger.warning("fp8lin: checksum mismatch, dropping a chunk")
                out.append(None)
            else:
                out.append(self._to_kv(c))
        return out

    # A corrupt blob decodes to None, which RemoteBackend.get_blocking reports as a
    # miss, hence the wider return type than the base class.
    def deserialize(  # type: ignore[override]
        self, memory_obj: MemoryObj
    ) -> Optional[MemoryObj]:
        """Decode one fp8lin blob (see :meth:`batched_deserialize`)."""
        return self.batched_deserialize([memory_obj])[0]
