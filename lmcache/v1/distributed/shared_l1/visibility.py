# SPDX-License-Identifier: Apache-2.0
"""Platform visibility ABI for shared Device-DAX ranges.

``MAP_SHARED``, ``msync``, or equal physical media alone are never accepted
as cross-host visibility proof. The operator supplies an absolute library
path exporting the qualified primitive; a missing library, missing symbol,
or failing call is fatal for the shared-L1 configuration — there is no
silent fallback.
"""

# Standard
from pathlib import Path
import ctypes
import os

PUBLISH = 1
"""Operation code that publishes one exact written range."""

ACQUIRE = 2
"""Operation code that acquires one exact range before reading."""

_VISIBILITY_SYMBOL = "lmcache_shared_l1_visibility_v1"


class NativeDeviceDaxVisibility:
    """Call the operator-supplied visibility implementation via ctypes.

    Args:
        library_path: Absolute path to the shared library exporting
            ``lmcache_shared_l1_visibility_v1``.

    Raises:
        ValueError: The path is not an absolute regular file.
        RuntimeError: The library does not provide the required symbol.
    """

    def __init__(self, library_path: str) -> None:
        path = Path(library_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "shared-L1 visibility library must be an absolute regular file"
            )
        try:
            library = ctypes.CDLL(str(path), use_errno=True)
            function = getattr(library, _VISIBILITY_SYMBOL)
        except (OSError, AttributeError) as exc:
            raise RuntimeError(f"{path} does not provide {_VISIBILITY_SYMBOL}") from exc
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

    @property
    def granularity(self) -> int:
        """The current ABI isolates x86 cache-line-sized ranges."""
        return 64

    def apply(
        self,
        operation: int,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        """Apply one publish/acquire without host-specific code in LMCache.

        Args:
            operation: ``PUBLISH`` or ``ACQUIRE``.
            device_fd: File descriptor of the open Device-DAX node.
            mapped_address: Local virtual address of the range start.
            device_offset: Byte offset of the range on the device.
            length: Exact byte length of the range.
            generation: The object's coordinator-assigned generation.

        Raises:
            RuntimeError: The native call returned a nonzero status.
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
            error_number = -int(result)
            detail = (
                os.strerror(error_number)
                if 0 < error_number < 4096
                else f"native status {result}"
            )
            raise RuntimeError(
                f"shared-L1 visibility operation {operation} failed: {detail}"
            )
