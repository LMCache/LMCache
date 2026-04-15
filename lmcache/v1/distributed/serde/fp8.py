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
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
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

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Cast src tensor to fp8 and copy bytes into dst buffer."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        # Cast to fp8 (1 byte per element)
        fp8_tensor = src_tensor.to(self._fp8_dtype).contiguous()
        n_bytes = fp8_tensor.numel()

        # Reinterpret fp8 bytes as uint8 and copy into dst byte buffer
        fp8_as_bytes = fp8_tensor.view(torch.uint8).flatten()
        dst_tensor.flatten()[:n_bytes].copy_(fp8_as_bytes)
        return n_bytes

    # Safety margin applied to the exact fp8 size. The actual serialized
    # output is exactly num_elements bytes; the 1.5x headroom absorbs
    # alignment padding or future format changes.
    _BUFFER_MARGIN: float = 1.5

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return buffer size for fp8 output (1 byte/elem * safety margin)."""
        total_elements = 0
        for shape in layout_desc.shapes:
            n = 1
            for dim in shape:
                n *= int(dim)
            total_elements += n
        return int(total_elements * self._BUFFER_MARGIN)


class Fp8QuantizationDeserializer(Deserializer):
    """Dequantize fp8 bytes back into the dst's original dtype."""

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self._fp8_dtype = fp8_dtype

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Read fp8 bytes from src, cast to dst's dtype, copy into dst."""
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("Fp8 serde requires src and dst to have tensors")

        n_elements = dst_tensor.numel()

        # Read n_elements bytes from src, reinterpret as fp8, reshape, cast back
        fp8_bytes = src_tensor.flatten()[:n_elements]
        fp8_tensor = fp8_bytes.view(self._fp8_dtype).reshape(dst_tensor.shape)
        dst_tensor.copy_(fp8_tensor.to(dst_tensor.dtype))
