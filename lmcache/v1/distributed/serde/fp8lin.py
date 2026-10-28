# SPDX-License-Identifier: Apache-2.0
"""
fp8lin serde: lossless predictive coding of FP8 KV chunks (multiprocess mode).

Each chunk is stored as E4M3 codes entropy coded on the GPU against a fitted linear
prediction (other heads of the same token, the token's previous layer); see
``lmcache/v1/fp8lin``. Typical size is 4.2-4.8 bits per value, 1.7-1.9x smaller than
FP8.

* uint8 / float8_e4m3fn KV (an FP8 KV cache) is stored **losslessly**.
* bf16 / fp16 KV is rounded to E4M3 at the coder's per-layer scales (lossy, like
  the ``fp8`` serde), then coded losslessly.

Variable-size output: ``serialize`` returns the bytes it wrote and the async
processor narrows the buffer to them; on load, the L2 adapter may return a shorter
object than the estimate (see ``FSL2Adapter``), and ``deserialize`` reads the blob's
own length from its header.

Config::

    "serde": {"type": "fp8lin", "coder_path": "/path/to/coder",
              "chunk_size": 256, "device": "cuda:0"}
"""

# Standard
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.fp8lin import ChunkCodec
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

FP8_STORAGE_DTYPES = (torch.uint8, torch.float8_e4m3fn)
E4M3_MAX = 448.0


class SharedCodec:
    """Codec, scales and lock shared by the serializer and the deserializer."""

    def __init__(self, coder_path: str, device: str, chunk_size: int):
        self.codec = ChunkCodec(coder_path, device)
        self.lock = threading.Lock()
        self.align = chunk_size
        s = self.codec.meta.get("kv_scale") or {}
        L = self.codec.L
        self.scale = torch.tensor(
            [s.get("k") or [1.0] * L, s.get("v") or [1.0] * L],
            dtype=torch.float32,
            device=self.codec.device,
        ).T[:, None, :, None, None]  # [L, 1, 2, 1, 1], matching kv_tensor views

    def kv_tensor(self, obj: MemoryObj) -> torch.Tensor:
        """An [L, T, 2, H, D] view (K, V heads of each token) of an L1 object's KV.

        Accepts [2, L, T, H*D] (separate K and V) and [1, L, T, H*2*D] (each head's K
        and V adjacent per token, as the vLLM MP connector stores them).

        Raises:
            ValueError: The object does not hold one KV group of this coder's shape.
        """
        t = obj.tensor
        L, H, D, C = self.codec.L, self.codec.H, self.codec.D, self.codec.C
        if t is None or t.dim() != 4 or t.shape[1] != L or t.shape[0] not in (1, 2):
            raise ValueError(
                f"fp8lin: unsupported KV layout {None if t is None else t.shape}"
            )
        if t.shape[3] != C * (3 - t.shape[0]):
            raise ValueError(
                f"fp8lin: hidden size {t.shape[3]} does not match coder {C}"
            )
        T = t.shape[2]
        if t.shape[0] == 2:
            return t.view(2, L, T, H, D).permute(1, 2, 0, 3, 4)
        return t[0].view(L, T, H, 2, D).permute(0, 1, 3, 2, 4)


class Fp8LinSerializer(Serializer):
    """Encode KV chunks into fp8lin blobs (one GPU pass per batch)."""

    def __init__(self, shared: SharedCodec):
        self._s = shared

    def _codes(self, kv: torch.Tensor) -> torch.Tensor:
        t = kv.to(self._s.codec.device, non_blocking=True)
        if t.dtype in FP8_STORAGE_DTYPES:
            c = t.view(torch.uint8)
        else:
            x = (t.float() / self._s.scale).clamp(-E4M3_MAX, E4M3_MAX)
            c = x.to(torch.float8_e4m3fn).view(torch.uint8)
        L, T = t.shape[:2]
        return c.reshape(L, T, -1).contiguous()

    def serialize_batch(
        self, srcs: list[MemoryObj], dsts: list[MemoryObj], keys: list[ObjectKey]
    ) -> list[int]:
        """Encode a batch of chunks; returns the bytes written into each dst.

        Raises:
            ValueError: A source has an unsupported layout or a destination is too
                small.
        """
        with self._s.lock:
            codes = [self._codes(self._s.kv_tensor(o)) for o in srcs]
            starts = self._s.codec.find_starts(codes, self._s.align)
            blobs = self._s.codec.encode(codes, starts)
        ns = []
        for blob, dst in zip(blobs, dsts, strict=True):
            d = dst.tensor
            if d is None or d.numel() * d.element_size() < len(blob):
                raise ValueError("fp8lin: destination buffer too small")
            d.view(torch.uint8).flatten()[: len(blob)].copy_(
                torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            )
            ns.append(len(blob))
        return ns

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encode one chunk into dst; returns the bytes written."""
        return self.serialize_batch([src], [dst], [key])[0]

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Header plus one byte per value: blobs are never larger than raw codes."""
        shape = layout_desc.shapes[0]
        return self._s.codec.max_blob_size(int(shape[2]))


class Fp8LinDeserializer(Deserializer):
    """Decode fp8lin blobs into KV chunks in the destination's dtype."""

    def __init__(self, shared: SharedCodec):
        self._s = shared

    def _blob(self, src: MemoryObj) -> bytes:
        t = src.tensor
        if t is None:
            raise ValueError("fp8lin: source has no tensor")
        raw = t.view(torch.uint8).flatten()
        head = bytes(raw[: self._s.codec.header_size()].cpu().numpy())
        n = self._s.codec.blob_length(head)
        if n > raw.numel():
            raise ValueError("fp8lin: truncated blob")
        return bytes(raw[:n].cpu().numpy())

    def deserialize_batch(
        self, srcs: list[MemoryObj], dsts: list[MemoryObj], keys: list[ObjectKey]
    ) -> None:
        """Decode a batch of blobs into the destinations.

        Raises:
            ValueError: A blob is malformed or fails its checksum (the load is then
                reported as failed, i.e. a cache miss).
        """
        blobs = [self._blob(o) for o in srcs]
        with self._s.lock:
            codes = self._s.codec.decode(blobs)
        for c, dst in zip(codes, dsts, strict=True):
            if c is None:
                raise ValueError("fp8lin: checksum mismatch")
            kv = self._s.kv_tensor(dst)
            k = c.view(kv.shape)
            if kv.dtype in FP8_STORAGE_DTYPES:
                kv.copy_(k.view(kv.dtype))
            else:
                kv.copy_(
                    (k.view(torch.float8_e4m3fn).float() * self._s.scale).to(kv.dtype)
                )

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decode one blob into dst."""
        self.deserialize_batch([src], [dst], [key])


def _create_fp8lin_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    path = kwargs.get("coder_path")
    if not path:
        raise ValueError("fp8lin serde requires 'coder_path'")
    dev = kwargs.get("device")
    device = str(dev) if dev else f"{torch_device_type}:{torch_dev.current_device()}"
    shared = SharedCodec(str(path), device, int(kwargs.get("chunk_size", 256)))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        Fp8LinSerializer(shared), Fp8LinDeserializer(shared), max_workers=1
    )


register_serde_factory("fp8lin", _create_fp8lin_serde)
