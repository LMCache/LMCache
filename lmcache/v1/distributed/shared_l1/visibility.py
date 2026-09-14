# SPDX-License-Identifier: Apache-2.0
"""Qualified Device-DAX publish/acquire ABI; no visibility fallback."""

# Standard
from pathlib import Path
import ctypes

PUBLISH = 1
ACQUIRE = 2


class NativeDeviceDaxVisibility:
    """Load a qualified visibility library from an absolute library_path.

    Invalid paths raise ValueError; native loader/symbol errors propagate.
    """

    granularity = 64

    def __init__(self, library_path: str) -> None:
        path = Path(library_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "shared-L1 visibility library must be an absolute regular file"
            )
        library = ctypes.CDLL(str(path))
        function = library.lmcache_shared_l1_visibility_v1
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        function.restype = ctypes.c_int
        self._library = library
        self._function = function

    def apply(
        self,
        operation: int,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        """Publish (1) or acquire (2) length bytes at the local mapped_address.

        Pass the open device_fd, absolute device_offset and handle generation
        to the synchronous ABI. Raise RuntimeError on any nonzero status.
        """
        result = self._function(
            b"software_fenced",
            operation,
            device_fd,
            ctypes.c_void_p(mapped_address),
            device_offset,
            length,
            generation,
        )
        if result:
            raise RuntimeError(
                f"shared-L1 visibility operation {operation} failed: {result}"
            )
