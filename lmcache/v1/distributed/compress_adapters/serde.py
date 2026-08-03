# SPDX-License-Identifier: Apache-2.0
"""Serializer/Deserializer implementations for accelerated KV compression."""

# Standard
from typing import cast

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.memory_management import MemoryObj


def _element_size_from_layout(layout_desc: MemoryLayoutDesc) -> int:
    """Infer element size in bytes from the layout descriptor."""
    if layout_desc.dtypes:
        return layout_desc.dtypes[0].itemsize
    return 2  # default bf16


class AccelCompressSerializer(Serializer):
    """Serializer that applies optional preprocessing then HW compression.

    Pipeline: [quant_trunc] -> [data_shuffle] -> compress

    Args:
        backend: Accelerated compression backend (e.g. QatBackend).
        byte_reorder: Whether to apply data_shuffle before compression.
        truncate_bits: Number of LSBs to zero (0 = disabled).
        element_size: Bytes per element (2 for bf16/fp16, 1 for fp8).
    """

    def __init__(
        self,
        backend: AccelCompressBackend,
        byte_reorder: bool = False,
        truncate_bits: int = 0,
        element_size: int = 2,
    ) -> None:
        self._backend = backend
        self._byte_reorder = byte_reorder
        self._truncate_bits = truncate_bits
        self._element_size = element_size

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Apply preprocessing then compress src into dst.

        Returns:
            Number of bytes written to dst.
        """
        src_buf = cast(memoryview, src.byte_array)
        dst_buf = cast(memoryview, dst.byte_array)

        # Preprocessing (in-place on src -- lossy for trunc)
        if self._truncate_bits > 0:
            self._backend.quant_trunc(src_buf, self._element_size, self._truncate_bits)
        if self._byte_reorder:
            self._backend.data_shuffle(src_buf, self._element_size)

        # Compress
        compressed_size = self._backend.compress(src_buf, dst_buf)
        return compressed_size

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Upper bound on compressed output size."""
        # Compute raw size from layout
        raw_size = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=False):
            numel = 1
            for dim in shape:
                numel *= dim
            raw_size += numel * dtype.itemsize
        return self._backend.max_compressed_length(raw_size)


class AccelCompressDeserializer(Deserializer):
    """Deserializer that decompresses then reverses preprocessing.

    Pipeline: decompress -> [data_shuffle (self-inverse)]

    Note: quant_trunc is lossy and NOT reversed on the deserialize path.

    Args:
        backend: Accelerated compression backend (e.g. QatBackend).
        byte_reorder: Whether data_shuffle was applied during serialization.
        element_size: Bytes per element (2 for bf16/fp16, 1 for fp8).
    """

    def __init__(
        self,
        backend: AccelCompressBackend,
        byte_reorder: bool = False,
        element_size: int = 2,
    ) -> None:
        self._backend = backend
        self._byte_reorder = byte_reorder
        self._element_size = element_size

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decompress src into dst, then reverse byte reorder if needed."""
        src_buf = cast(memoryview, src.byte_array)
        dst_buf = cast(memoryview, dst.byte_array)

        # Decompress
        self._backend.decompress(src_buf, dst_buf)

        # Reverse preprocessing (data_shuffle is self-inverse)
        if self._byte_reorder:
            self._backend.data_shuffle(dst_buf, self._element_size)
