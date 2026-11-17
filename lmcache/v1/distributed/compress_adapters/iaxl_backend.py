# SPDX-License-Identifier: Apache-2.0
"""Compression backend using the kv_zip codec of intel-accel-for-llm (iaxl)."""

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend


class IaxlBackend(AccelCompressBackend):
    """Compresses on iaxl's QAT, IAA or CPU workers.

    Each block becomes one self-describing kv_zip block. iaxl splits blocks
    into segments and applies data shuffling and lossy truncation itself, as
    its ``IAXL_*`` environment variables select. Set
    ``IAXL_ZIP_SEGMENT_SIZE_KB`` when objects exceed ``IAXL_ZIP_SRC_CAP``.

    Args:
        dtype: Element type of the KV data. It selects BF16 data shuffling
            and the lossy truncation mask.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        try:
            # Third Party
            from iaxl import torch_ext
        except ImportError as e:
            raise ImportError(
                "accel_kv_compress backend 'iaxl' requires the iaxl package"
            ) from e
        self._ext = torch_ext
        self._dtype = dtype

    def compress(self, src: memoryview, dst: memoryview) -> int:
        return self.compress_blocks([src], [dst])[0]

    def decompress(self, src: memoryview, dst: memoryview) -> int:
        return self.decompress_blocks([src], [dst])[0]

    def max_compressed_length(self, src_size: int) -> int:
        return self._ext.zip_max_block_size(src_size)

    def compress_blocks(
        self, srcs: Sequence[memoryview], dsts: Sequence[memoryview]
    ) -> list[int]:
        return self._ext.zip_compress_into(list(srcs), list(dsts), self._dtype)

    def decompress_blocks(
        self, srcs: Sequence[memoryview], dsts: Sequence[memoryview]
    ) -> list[int]:
        return self._ext.zip_decompress_into(list(srcs), list(dsts))

    def close(self) -> None:
        pass
