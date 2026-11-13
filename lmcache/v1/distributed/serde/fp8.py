# SPDX-License-Identifier: Apache-2.0
"""
Simple fp8 quantization serde.

Casts KV cache tensors to fp8 (1 byte per element) on serialize, and
casts back to the destination's original dtype on deserialize.

Lossy: precision below fp8's representable range is lost.

When ``scaled=True``, a fused amax + scale + FP8 cast kernel writes a per-group
scale alongside the fp8 bytes so the original magnitude is restored on
deserialize instead of the normalized (0..448) range.
"""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj

# Size (in bytes) of the u32 scale-count prefix written before the scale table.
_SCALE_COUNT_BYTES = 4


def _n_scales(n: int, block_size: int, cols: int) -> int:
    """Number of scale groups for a tensor of ``n`` elements and last-dim ``cols``.

    Mirrors the granularity selection in :func:`_fused_scaled_cast` so the
    serializer and deserializer agree on how many scale floats to read back.
    """
    if block_size <= 0 or n % block_size != 0:
        # Rowwise fallback: one scale per row (one x.size(-1) slice).
        return max(1, n // max(1, cols))
    # Blockwise: one scale per block_size elements.
    return n // block_size


def _fused_scaled_cast(
    t: torch.Tensor,
    fp8_dtype: torch.dtype,
    block_size: int,
    amax_ceiling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-pass amax + scale + fp8 cast via the fused CUDA kernel.

    Returns ``(fp8_tensor, scales)`` where ``scales`` holds the dequant
    multiplier (amax / 448.0) per group, so ``x ~= fp8 * scales`` elementwise.

    Falls back to the scale-free path (returning ``scales=None``) if the
    extension is unavailable, the tensor is not on CUDA, or the dtype is
    unsupported, so this is always safe to call.
    """
    try:
        from lmcache import cuda_ops
    except Exception:
        return t.to(fp8_dtype).contiguous(), None

    if not t.is_cuda or t.dtype not in (torch.float32, torch.bfloat16):
        return t.to(fp8_dtype).contiguous(), None

    x = t.contiguous().to(torch.float32)
    n = x.numel()
    cols = x.size(-1)
    if block_size <= 0 or n % block_size != 0:
        scale_mode, n_scales = 1, _n_scales(n, block_size, cols)
    else:
        scale_mode, n_scales = 2, _n_scales(n, block_size, cols)

    fp8 = torch.empty_like(x, dtype=fp8_dtype)
    scales = torch.empty(n_scales, dtype=torch.float32, device=x.device)
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, scale_mode, block_size, amax_ceiling)
    return fp8, scales


class Fp8QuantizationSerializer(Serializer):
    """Quantize KV cache tensors to fp8 for L2 storage.

    Args:
        fp8_dtype: torch fp8 dtype to use. Defaults to float8_e4m3fn
            (4-bit exponent, 3-bit mantissa, finite-only — good range
            for inference activations).
        scaled: if True, use the fused amax+scale+cast kernel instead of a
            scale-free ``.to(fp8)``. A scale-free cast saturates at +-448
            regardless of the tensor's amax, which discards most of the
            representable range when the tensor is small. Default False so
            existing behaviour is unchanged.
        block_size: elements per scale group when scaled=True and the tensor
            is quantized per-block (the default). 128 matches torchao /
            FlashInfer.
        amax_ceiling: if > 0, the scale is computed from
            max(tensor_amax, amax_ceiling) so dequantized values stay
            comparable across tensors.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scaled: bool = False,
        block_size: int = 128,
        amax_ceiling: float = 0.0,
    ):
        self._fp8_dtype = fp8_dtype
        self._scaled = scaled
        self._block_size = block_size
        self._amax_ceiling = amax_ceiling

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Serialize src KV data into dst byte buffer (in-place).

        Scale-free (default): writes exactly ``n_elements`` fp8 bytes — the
        same format as before, so existing stored data keeps round-tripping.

        Scaled: writes a self-describing layout
        ``[u32 n_scales][scales (f32 x n_scales)][fp8 bytes]`` so the
        matching scaled deserializer can restore the original magnitude.
        """
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        n_elements = src_tensor.numel()

        if self._scaled and src_tensor.is_cuda:
            fp8_tensor, scales = _fused_scaled_cast(
                src_tensor, self._fp8_dtype, self._block_size, self._amax_ceiling
            )
            if scales is not None:
                # Kernel path: [u32 n_scales][scales f32][fp8 bytes]
                n_scales = scales.numel()
                n_bytes = _SCALE_COUNT_BYTES + 4 * n_scales + n_elements

                count_buf = torch.tensor(
                    [n_scales], dtype=torch.uint32, device=dst_tensor.device
                )
                dst_tensor.flatten()[0:_SCALE_COUNT_BYTES].copy_(
                    count_buf.view(torch.uint8)
                )
                scales_bytes = scales.view(torch.uint8).flatten()
                dst_tensor.flatten()[
                    _SCALE_COUNT_BYTES : _SCALE_COUNT_BYTES + 4 * n_scales
                ].copy_(scales_bytes)
                fp8_bytes = fp8_tensor.view(torch.uint8).flatten()
                fp8_start = _SCALE_COUNT_BYTES + 4 * n_scales
                dst_tensor.flatten()[fp8_start : fp8_start + n_elements].copy_(
                    fp8_bytes
                )
                return n_bytes
            # scales is None: kernel unavailable / dtype unsupported ->
            # fall through to the fallback below.

        # Fallback / scale-free path.
        fp8_tensor = src_tensor.to(self._fp8_dtype).contiguous()
        fp8_as_bytes = fp8_tensor.view(torch.uint8).flatten()
        if self._scaled:
            # Keep the scaled byte layout ([u32 count][fp8 bytes]) with
            # count=0 so a scaled deserializer can read it (e.g. CPU without
            # the CUDA kernel, or an unsupported dtype).
            count_buf = torch.tensor(
                [0], dtype=torch.uint32, device=dst_tensor.device
            )
            dst_tensor.flatten()[0:_SCALE_COUNT_BYTES].copy_(
                count_buf.view(torch.uint8)
            )
            dst_tensor.flatten()[
                _SCALE_COUNT_BYTES : _SCALE_COUNT_BYTES + n_elements
            ].copy_(fp8_as_bytes)
            return _SCALE_COUNT_BYTES + n_elements

        # Cast to fp8 (1 byte per element), scale-free
        dst_tensor.flatten()[:n_elements].copy_(fp8_as_bytes)
        return n_elements

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return buffer size for fp8 output.

        Scale-free: exactly 1 byte per element (unchanged).
        Scaled: ``[u32 n_scales] + scales(f32 x n_scales) + fp8 bytes``.
        """
        total_elements = 0
        for shape in layout_desc.shapes:
            n = 1
            for dim in shape:
                n *= int(dim)
            total_elements += n

        if not self._scaled:
            return total_elements

        cols = int(layout_desc.shapes[-1][-1]) if layout_desc.shapes else 1
        n_scales = _n_scales(total_elements, self._block_size, cols)
        return _SCALE_COUNT_BYTES + 4 * n_scales + total_elements


class Fp8QuantizationDeserializer(Deserializer):
    """Dequantize fp8 bytes back into the dst's original dtype.

    When constructed with ``scaled=True``, reads the scale table written by
    the matching scaled serializer and restores the original magnitude
    (``x ~= fp8 * scales``). When ``scaled=False`` (default), reads the
    legacy scale-free byte layout unchanged.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scaled: bool = False,
        block_size: int = 128,
    ):
        self._fp8_dtype = fp8_dtype
        self._scaled = scaled
        self._block_size = block_size

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Read fp8 bytes from src, optionally apply scales, copy into dst."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        n_elements = dst_tensor.numel()
        flat = src_tensor.flatten()

        if not self._scaled:
            # Legacy scale-free layout: [fp8 bytes]
            fp8_bytes = flat[:n_elements]
            fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
            dst_tensor.copy_(fp8_tensor.to(dst_tensor.dtype))
            return

        # Scaled layout: [u32 n_scales][scales f32 x n_scales][fp8 bytes]
        n_scales = int(flat[0:_SCALE_COUNT_BYTES].view(torch.uint32).item())
        if n_scales == 0:
            # Scale-free fallback (e.g. CPU without the CUDA kernel): the
            # payload starts right after the count prefix, no scales to apply.
            fp8_bytes = flat[
                _SCALE_COUNT_BYTES : _SCALE_COUNT_BYTES + n_elements
            ]
            fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
            dst_tensor.copy_(fp8_tensor.to(dst_tensor.dtype))
            return

        scales = flat[
            _SCALE_COUNT_BYTES : _SCALE_COUNT_BYTES + 4 * n_scales
        ].view(torch.float32)
        fp8_bytes = flat[
            _SCALE_COUNT_BYTES + 4 * n_scales : _SCALE_COUNT_BYTES
            + 4 * n_scales
            + n_elements
        ]
        fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
        dequant = fp8_tensor.to(torch.float32)

        # Reapply the per-group dequant multiplier using the same granularity
        # the serializer used (determined from block_size + shape).
        n = dequant.numel()
        cols = int(dst_tensor.shape[-1]) if dst_tensor.ndim > 0 else 1
        if self._block_size > 0 and n % self._block_size == 0:
            # Blockwise: one scale per block_size elements.
            n_groups = n // self._block_size
            dequant = (
                dequant.reshape(n_groups, self._block_size)
                * scales.reshape(n_groups, 1)
            ).reshape(dst_tensor.shape)
        else:
            # Rowwise: one scale per x.size(-1) slice.
            rows = n // max(1, cols)
            dequant = (
                dequant.reshape(rows, cols) * scales.reshape(rows, 1)
            ).reshape(dst_tensor.shape)

        dst_tensor.copy_(dequant.to(dst_tensor.dtype))


def _create_fp8_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    dtype_name = str(kwargs.get("fp8_dtype", "float8_e4m3fn"))
    fp8_dtype = getattr(torch, dtype_name, None)
    if fp8_dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}")

    scaled = bool(kwargs.get("scaled", False))
    block_size = int(kwargs.get("block_size", 128))
    amax_ceiling = float(kwargs.get("amax_ceiling", 0.0))

    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        Fp8QuantizationSerializer(
            fp8_dtype, scaled=scaled, block_size=block_size,
            amax_ceiling=amax_ceiling,
        ),
        Fp8QuantizationDeserializer(
            fp8_dtype, scaled=scaled, block_size=block_size,
        ),
        max_workers=max_workers,
    )


register_serde_factory("fp8", _create_fp8_serde)
