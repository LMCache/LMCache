# SPDX-License-Identifier: Apache-2.0
"""
Simple fp8 quantization serde.

Casts KV cache tensors to fp8 (1 byte per element) on serialize, and
casts back to the destination's original dtype on deserialize.

Lossy: precision below fp8's representable range is lost.
"""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj


class Fp8QuantizationSerializer(Serializer):
    """Quantize KV cache tensors to fp8 for L2 storage.

    Args:
        fp8_dtype: torch fp8 dtype to use. Defaults to float8_e4m3fn
            (4-bit exponent, 3-bit mantissa, finite-only — good range
            for inference activations).
    """

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self._fp8_dtype = fp8_dtype

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Cast src tensor to fp8 directly into dst's byte buffer (key unused)."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        # fp8 is 1 byte per element. copy_ casts while writing into an fp8
        # view of dst, so no intermediate fp8 tensor is allocated.
        n_bytes = src_tensor.numel()
        dst_fp8 = dst_tensor.flatten()[:n_bytes].view(self._fp8_dtype)
        dst_fp8.view(src_tensor.shape).copy_(src_tensor)
        return n_bytes

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return buffer size for fp8 output: exactly 1 byte per element.

        fp8 has a fixed 1:1 element-to-byte mapping, so the size is
        deterministic — no margin is needed. Inflating the estimate
        would inflate the bytes the wrapped L2 adapter persists (it
        stores the whole MemoryObj), eroding the storage savings fp8
        is meant to provide.
        """
        total_elements = 0
        for shape in layout_desc.shapes:
            n = 1
            for dim in shape:
                n *= int(dim)
            total_elements += n
        return total_elements


class Fp8QuantizationDeserializer(Deserializer):
    """Dequantize fp8 bytes back into the dst's original dtype."""

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self._fp8_dtype = fp8_dtype

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Read fp8 bytes from src, cast to dst's dtype, copy into dst (key unused)."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        n_elements = dst_tensor.numel()

        # Read n_elements bytes from src and reinterpret them as fp8. copy_
        # casts to dst's dtype while writing, so no intermediate tensor in
        # dst's dtype is allocated.
        fp8_bytes = src_tensor.flatten()[:n_elements]
        fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
        dst_tensor.copy_(fp8_tensor)


def _create_fp8_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    dtype_name = str(kwargs.get("fp8_dtype", "float8_e4m3fn"))
    fp8_dtype = getattr(torch, dtype_name, None)
    if fp8_dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}")

    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        Fp8QuantizationSerializer(fp8_dtype),
        Fp8QuantizationDeserializer(fp8_dtype),
        max_workers=max_workers,
    )


register_serde_factory("fp8", _create_fp8_serde)
