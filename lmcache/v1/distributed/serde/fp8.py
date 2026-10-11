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
        """Cast src tensor to fp8 and copy bytes into dst buffer (key unused)."""
        if src.tensor is None or dst.tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        # Check if using separate-object-groups layout (has per-group shapes/dtypes)
        if src.meta.shapes is not None and src.meta.dtypes is not None:
            # Iterate over groups when using separate-object-groups layout
            n_bytes = 0
            for i in range(len(src.meta.shapes)):
                group_tensor = src.get_tensor(i)
                if group_tensor is None:
                    continue
                # Cast to fp8 (1 byte per element)
                fp8_tensor = group_tensor.to(self._fp8_dtype).contiguous()
                n_bytes += fp8_tensor.numel()

                # Reinterpret fp8 bytes as uint8 and copy into dst byte buffer
                fp8_as_bytes = fp8_tensor.view(torch.uint8).flatten()
                # Compute dst offset for this group
                dst_group_tensor = dst.get_tensor(i)
                if dst_group_tensor is not None:
                    dst_group_tensor.flatten()[: fp8_tensor.numel()].copy_(
                        fp8_as_bytes[: fp8_tensor.numel()]
                    )
            return n_bytes

        # Legacy single-group path (fallback for backward compatibility)
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        # Cast to fp8 (1 byte per element)
        fp8_tensor = src_tensor.to(self._fp8_dtype).contiguous()
        n_bytes = fp8_tensor.numel()

        # Reinterpret fp8 bytes as uint8 and copy into dst byte buffer
        fp8_as_bytes = fp8_tensor.view(torch.uint8).flatten()
        dst_tensor.flatten()[:n_bytes].copy_(fp8_as_bytes)
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
        if src.tensor is None or dst.tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        # Check if using separate-object-groups layout (has per-group shapes/dtypes)
        if (
            src.meta.shapes is not None
            and src.meta.dtypes is not None
            and dst.meta.shapes is not None
            and dst.meta.dtypes is not None
        ):
            # Iterate over groups when using separate-object-groups layout
            for i in range(len(dst.meta.shapes)):
                dst_group_tensor = dst.get_tensor(i)
                if dst_group_tensor is None:
                    continue
                n_elements = dst_group_tensor.numel()

                # Read n_elements bytes from src for this group, reinterpret as fp8,
                # reshape, cast back, and copy into dst group
                src_group_tensor = src.get_tensor(i)
                if src_group_tensor is None:
                    continue
                fp8_bytes = src_group_tensor.flatten()[:n_elements]
                fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(
                    dst_group_tensor.shape
                )
                dst_group_tensor.copy_(fp8_tensor.to(dst_group_tensor.dtype))
        else:
            # Legacy single-group path (fallback for backward compatibility)
            src_tensor = src.tensor
            dst_tensor = dst.tensor
            n_elements = dst_tensor.numel()

            # Read n_elements bytes from src, reinterpret as fp8, reshape, cast back
            fp8_bytes = src_tensor.flatten()[:n_elements]
            fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
            dst_tensor.copy_(fp8_tensor.to(dst_tensor.dtype))


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
