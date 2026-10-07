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




def _fused_scaled_cast(
    t: torch.Tensor,
    fp8_dtype: torch.dtype,
    block_size: int,
    amax_ceiling: float,
) -> torch.Tensor:
    """One-pass amax + scale + fp8 cast via the fused CUDA kernel.

    Falls back to the scale-free path if the extension is unavailable or the
    tensor is not a supported dtype, so this is always safe to call.
    """
    try:
        from lmcache import cuda_ops
    except Exception:
        return t.to(fp8_dtype).contiguous()

    if t.dtype not in (torch.float32, torch.bfloat16):
        return t.to(fp8_dtype).contiguous()

    x = t.contiguous()
    n = x.numel()
    # blockwise needs at least one full group
    if block_size <= 0 or n % block_size != 0:
        scale_mode, n_scales = 1, max(1, n // max(1, x.size(-1)))
    else:
        scale_mode, n_scales = 2, n // block_size

    fp8 = torch.empty_like(x, dtype=fp8_dtype)
    scales = torch.empty(n_scales, dtype=torch.float32, device=x.device)
    cuda_ops.fp8_quantize_scaled(
        x.contiguous().to(torch.float32), fp8, scales,
        scale_mode, block_size, amax_ceiling,
    )
    return fp8


class Fp8QuantizationSerializer(Serializer):
    """Quantize KV cache tensors to fp8 for L2 storage.

    Args:
        fp8_dtype: torch fp8 dtype to use. Defaults to float8_e4m3fn
            (4-bit exponent, 3-bit mantissa, finite-only — good range
            for inference activations).
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scaled: bool = False,
        block_size: int = 128,
        amax_ceiling: float = 0.0,
    ):
        """
        Args:
            fp8_dtype: torch fp8 dtype to use. Defaults to float8_e4m3fn
                (4-bit exponent, 3-bit mantissa, finite-only — good range
                for inference activations).
            scaled: if True, use the fused amax+scale+cast kernel instead of a
                scale-free `.to(fp8)`. A scale-free cast saturates at ±448
                regardless of the tensor's amax, which discards most of the
                representable range when the tensor is small. Default False so
                existing behaviour is unchanged.
            block_size: elements per scale group when scaled=True and the
                tensor is quantized per-block (the default). 128 matches
                torchao / FlashInfer.
            amax_ceiling: if > 0, the scale is computed from
                max(tensor_amax, amax_ceiling) so dequantized values stay
                comparable across tensors.
        """
        self._fp8_dtype = fp8_dtype
        self._scaled = scaled
        self._block_size = block_size
        self._amax_ceiling = amax_ceiling

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Cast src tensor to fp8 and copy bytes into dst buffer (key unused)."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        if self._scaled and src_tensor.is_cuda:
            fp8_tensor = _fused_scaled_cast(src_tensor, self._fp8_dtype,
                                            self._block_size,
                                            self._amax_ceiling)
        else:
            # Cast to fp8 (1 byte per element), scale-free
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
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

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
