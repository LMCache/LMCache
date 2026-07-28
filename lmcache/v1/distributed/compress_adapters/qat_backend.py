# SPDX-License-Identifier: Apache-2.0
"""QAT compression backend using KVCacheClip's libkvclip_qzip.so via ctypes."""

# Standard
import ctypes
import os

# First Party
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend

# The QZIP_SPLIT_BUF_SIZE from qzip.h (max single-block input size)
_QZIP_SPLIT_BUF_SIZE = 32 * 64 * 1024 // 2  # 1 MB


def _load_qzip_library(lib_path: str | None = None) -> ctypes.CDLL:
    """Load the libkvclip_qzip shared library.

    Resolution order:
    1. Explicit lib_path argument
    2. KVCLIP_QZIP_LIB_PATH environment variable
    3. Default linker search (LD_LIBRARY_PATH)

    Args:
        lib_path: Optional explicit path to the .so file.

    Returns:
        Loaded ctypes.CDLL handle.

    Raises:
        OSError: If the library cannot be found or loaded.
    """
    if lib_path is None:
        lib_path = os.environ.get("KVCLIP_QZIP_LIB_PATH", "libkvclip_qzip.so")
    lib = ctypes.CDLL(lib_path)

    # int kv_agent_block_compress(char*[], char*[], int[], int[], int)
    lib.kv_agent_block_compress.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),  # inputs[]
        ctypes.POINTER(ctypes.c_char_p),  # outputs[]
        ctypes.POINTER(ctypes.c_int),  # in_data_sizes[]
        ctypes.POINTER(ctypes.c_int),  # out_data_sizes[]
        ctypes.c_int,  # num
    ]
    lib.kv_agent_block_compress.restype = ctypes.c_int

    # int kv_agent_block_decompress(char*[], char*[], int[], int[], int)
    lib.kv_agent_block_decompress.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.kv_agent_block_decompress.restype = ctypes.c_int

    return lib


def _addr_to_c_char_p(addr: int) -> ctypes.c_char_p:
    """Convert an integer address to a c_char_p without copying."""
    return ctypes.cast(addr, ctypes.c_char_p)


class QatBackend(AccelCompressBackend):
    """QAT compression backend via KVCacheClip's libkvclip_qzip.so.

    Thread safety: The C library manages per-thread QAT sessions
    internally (__thread QzipThreadSession_T), so concurrent calls from
    different Python threads (via AsyncSerdeProcessor's ThreadPoolExecutor)
    are safe without additional locking.

    Args:
        lib_path: Optional explicit path to libkvclip_qzip.so.
    """

    def __init__(self, lib_path: str | None = None) -> None:
        self._lib = _load_qzip_library(lib_path)

    def compress(self, src: memoryview, dst: memoryview) -> int:
        """Compress src into dst using QAT hardware acceleration.

        For inputs larger than QZIP_SPLIT_BUF_SIZE (1 MB), the data is
        split into multiple blocks and compressed in a single batch call.

        Returns:
            Total number of bytes written to dst.
        """
        src_len = len(src)
        dst_len = len(dst)

        # Split into blocks of at most _QZIP_SPLIT_BUF_SIZE
        num_blocks = (src_len + _QZIP_SPLIT_BUF_SIZE - 1) // _QZIP_SPLIT_BUF_SIZE

        # Get base addresses from memoryviews
        src_buf = (ctypes.c_char * src_len).from_buffer(src)
        dst_buf = (ctypes.c_char * dst_len).from_buffer(dst)
        src_addr = ctypes.addressof(src_buf)
        dst_addr = ctypes.addressof(dst_buf)

        # Build arrays for batch API
        inputs_arr = (ctypes.c_char_p * num_blocks)()
        outputs_arr = (ctypes.c_char_p * num_blocks)()
        in_sizes_arr = (ctypes.c_int * num_blocks)()
        out_sizes_arr = (ctypes.c_int * num_blocks)()

        # Compute per-block output capacity (proportional split of dst)
        out_capacity_per_block = dst_len // num_blocks

        offset = 0
        out_offset = 0
        for i in range(num_blocks):
            block_size = min(_QZIP_SPLIT_BUF_SIZE, src_len - offset)
            inputs_arr[i] = _addr_to_c_char_p(src_addr + offset)
            outputs_arr[i] = _addr_to_c_char_p(dst_addr + out_offset)
            in_sizes_arr[i] = block_size
            out_sizes_arr[i] = (
                out_capacity_per_block if i < num_blocks - 1 else (dst_len - out_offset)
            )
            offset += block_size
            out_offset += out_capacity_per_block

        ret = self._lib.kv_agent_block_compress(
            inputs_arr,
            outputs_arr,
            in_sizes_arr,
            out_sizes_arr,
            ctypes.c_int(num_blocks),
        )
        if ret != 0:
            raise RuntimeError(f"kv_agent_block_compress failed with code {ret}")

        # Pack compressed blocks contiguously in dst.
        # The library wrote each block at a fixed offset (out_capacity_per_block
        # apart), but actual compressed sizes are smaller — leaving gaps.
        # We must compact so the caller can store dst[:total] as one blob.
        write_offset = 0
        for i in range(num_blocks):
            actual_size = out_sizes_arr[i]
            block_start = i * out_capacity_per_block
            if block_start != write_offset:
                ctypes.memmove(
                    dst_addr + write_offset,
                    dst_addr + block_start,
                    actual_size,
                )
            write_offset += actual_size

        return write_offset

    def decompress(self, src: memoryview, dst: memoryview) -> int:
        """Decompress src into dst using QAT hardware acceleration.

        Returns:
            Total number of bytes written to dst.
        """
        src_len = len(src)
        dst_len = len(dst)

        src_buf = (ctypes.c_char * src_len).from_buffer(src)
        dst_buf = (ctypes.c_char * dst_len).from_buffer(dst)
        src_addr = ctypes.addressof(src_buf)
        dst_addr = ctypes.addressof(dst_buf)

        inputs_arr = (ctypes.c_char_p * 1)(_addr_to_c_char_p(src_addr))
        outputs_arr = (ctypes.c_char_p * 1)(_addr_to_c_char_p(dst_addr))
        in_sizes_arr = (ctypes.c_int * 1)(src_len)
        out_sizes_arr = (ctypes.c_int * 1)(dst_len)

        ret = self._lib.kv_agent_block_decompress(
            inputs_arr,
            outputs_arr,
            in_sizes_arr,
            out_sizes_arr,
            ctypes.c_int(1),
        )
        if ret != 0:
            raise RuntimeError(f"kv_agent_block_decompress failed with code {ret}")

        return out_sizes_arr[0]

    def max_compressed_length(self, src_size: int) -> int:
        """Return worst-case compressed size (150% of input, matching KVCacheClip)."""
        return src_size * 3 // 2

    def close(self) -> None:
        """No-op: per-thread sessions are managed by the C library."""
        pass
