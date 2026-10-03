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


def _group_range(
    memory_obj: MemoryObj, index: int
) -> tuple[int, int, torch.dtype, torch.Size]:
    """Byte span, dtype, and shape of one group."""
    shapes = memory_obj.get_shapes()
    dtypes = memory_obj.get_dtypes()
    begin = 0
    for shape, dtype in zip(shapes[:index], dtypes[:index], strict=True):
        begin += shape.numel() * dtype.itemsize
    shape = shapes[index]
    dtype = dtypes[index]
    return begin, begin + shape.numel() * dtype.itemsize, dtype, shape


def _read_group(memory_obj: MemoryObj, index: int) -> torch.Tensor:
    """Read one group. Copy bytes when the offset cannot be viewed in place."""
    begin, end, dtype, shape = _group_range(memory_obj, index)
    if begin % dtype.itemsize == 0:
        group = memory_obj.get_tensor(index)
        if group is None:
            raise ValueError("Fp8 serde requires every logical group to have a tensor")
        return group
    return memory_obj.raw_data[begin:end].clone().view(dtype).view(shape)


def _write_group(memory_obj: MemoryObj, index: int, values: torch.Tensor) -> None:
    """Write one group back into the allocation, not into a read copy."""
    begin, end, dtype, _shape = _group_range(memory_obj, index)
    if begin % dtype.itemsize == 0:
        group = memory_obj.get_tensor(index)
        if group is None:
            raise ValueError("Fp8 serde requires every logical group to have a tensor")
        group.copy_(values)
        return
    as_bytes = values.contiguous().view(torch.uint8).reshape(-1)
    memory_obj.raw_data[begin:end].copy_(as_bytes)


def _kv_groups(memory_obj: MemoryObj) -> list[torch.Tensor]:
    """Groups in metadata order. A single tensor stays on ``.tensor``."""
    get_shapes = getattr(memory_obj, "get_shapes", None)
    shapes: list[torch.Size] | None = None
    if callable(get_shapes):
        try:
            shapes = list(get_shapes())
        except (AssertionError, NotImplementedError):
            shapes = None
    if shapes is None or len(shapes) <= 1:
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")
        return [tensor]
    return [_read_group(memory_obj, index) for index in range(len(shapes))]


def _flat_bytes(memory_obj: MemoryObj) -> torch.Tensor:
    """Byte view of ``tensor``, including a ``set_used_size`` narrow."""
    tensor = memory_obj.tensor
    if tensor is None:
        raise ValueError("Fp8 serde requires src and dst to have tensors")
    return tensor.flatten()


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
        """Quantize each source group to fp8 and pack the bytes into ``dst``.

        Args:
            src: KV object to quantize.
            dst: Writable byte buffer, at least one byte per source element.
            key: Unused. Present for the serializer interface.

        Returns:
            Bytes written. One byte per source element.

        Raises:
            ValueError: If a tensor is missing, the fp8 dtype is not one
                byte per element, or ``dst`` is too small.
        """
        del key
        groups = _kv_groups(src)
        dst_flat = _flat_bytes(dst)
        # Convert first so a short dst or a wide dtype leaves dst unchanged.
        packed_groups: list[torch.Tensor] = []
        n_bytes = 0
        for group in groups:
            packed = group.to(self._fp8_dtype).contiguous().view(torch.uint8).flatten()
            if packed.numel() != group.numel():
                raise ValueError(
                    "Fp8 serde requires a 1-byte dtype, got "
                    f"{self._fp8_dtype} ({packed.numel()} bytes for "
                    f"{group.numel()} elements)"
                )
            packed_groups.append(packed)
            n_bytes += packed.numel()
        if dst_flat.numel() < n_bytes:
            raise ValueError(
                f"Fp8 destination buffer too small: got {dst_flat.numel()} "
                f"bytes, need {n_bytes}"
            )

        offset = 0
        for packed in packed_groups:
            dst_flat[offset : offset + packed.numel()].copy_(packed)
            offset += packed.numel()
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
        """Restore each destination group from packed fp8 bytes.

        Args:
            src: Packed fp8 byte buffer.
            dst: KV object that receives the restored values.
            key: Unused. Present for the deserializer interface.

        Raises:
            ValueError: If a tensor is missing, or ``src`` has fewer
                bytes than ``dst`` has elements.
        """
        del key
        groups = _kv_groups(dst)
        src_flat = _flat_bytes(src)
        n_elements = sum(group.numel() for group in groups)
        if src_flat.numel() < n_elements:
            raise ValueError(
                f"Fp8 source buffer too small: got {src_flat.numel()} "
                f"bytes, need {n_elements}"
            )

        offset = 0
        multi_group = len(groups) > 1
        for index, group in enumerate(groups):
            n = group.numel()
            restored = (
                src_flat[offset : offset + n]
                .view(self._fp8_dtype)
                .reshape(group.shape)
                .to(dtype=group.dtype)
            )
            if multi_group:
                _write_group(dst, index, restored)
            else:
                group.copy_(restored)
            offset += n


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
