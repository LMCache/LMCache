# SPDX-License-Identifier: Apache-2.0
"""ROCm memory pinning via ``libamdhip64.so``.

On ROCm, the CUDA runtime library (``libcudart.so``) does not exist.
Instead, ROCm ships ``libamdhip64.so``, which exports the HIP-compatible
``hipHostRegister`` / ``hipHostUnregister`` symbols with the same C
signatures as their CUDA counterparts:

.. code-block:: c

    hipError_t hipHostRegister(void *ptr, size_t size, unsigned int flags);
    hipError_t hipHostUnregister(void *ptr);

The flag values are identical (``hipHostRegisterDefault = 0``,
``hipHostRegisterMapped = 0x2``).

This backend loads ``libamdhip64.so`` via :mod:`ctypes` and binds the
HIP pinning symbols.  It is selected automatically on ROCm via
:class:`RocmDeviceSpec.pin_memory_backend`.
"""

# Standard
import ctypes
import ctypes.util

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend

logger = init_logger(__name__)


def _load_libamdhip64() -> ctypes.CDLL | None:
    """Try to load ``libamdhip64`` and bind the HIP pinning symbols.

    The loaded library is configured with the
    ``hipHostRegister(void*, size_t, unsigned int)`` and
    ``hipHostUnregister(void*)`` signatures expected by the backend.

    Returns:
        The loaded ``ctypes.CDLL`` library with bound symbols on success,
        or ``None`` if the library cannot be found or loaded.
    """
    # Try several names: find_library first, then hardcoded fallbacks.
    found = ctypes.util.find_library("amdhip64")
    if found is not None:
        candidates = [found]
    else:
        candidates = []
    candidates.extend(["libamdhip64.so", "libamdhip64.so.1"])
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            lib.hipHostRegister.restype = ctypes.c_int
            lib.hipHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.hipHostUnregister.restype = ctypes.c_int
            lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
            logger.info("RocmPinMemoryBackend: loaded %s", path)
            return lib
        except (AttributeError, OSError) as exc:
            logger.debug(
                "RocmPinMemoryBackend: failed to load %s: %s", path, exc
            )
            continue

    return None


class RocmPinMemoryBackend(PinMemoryBackend):
    """ROCm memory pinning backend using ``libamdhip64.so``.

    Pinning loads ``libamdhip64.so`` via :mod:`ctypes` and calls
    ``hipHostRegister`` / ``hipHostUnregister``.  These have identical
    C signatures and flag values to their CUDA counterparts, so callers
    (e.g. ``current_device_spec.pin_memory(ptr, size, flags)``) need no
    changes.

    Attributes:
        _libhip: ``ctypes``-loaded ROCm runtime when
            ``libamdhip64.so`` is available.
    """

    def __init__(self) -> None:
        """Initialize the backend by loading ``libamdhip64.so``.

        If the library cannot be found, the backend stays in an
        unsupported state and ``is_pin_supported`` returns ``False``.
        """
        self._libhip = _load_libamdhip64()
        if self._libhip is not None:
            logger.info("RocmPinMemoryBackend: using libamdhip64 via ctypes")
        else:
            logger.warning(
                "RocmPinMemoryBackend: libamdhip64.so not found; "
                "pinning will be a no-op (slower H2D/D2H copies)"
            )

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region using ``hipHostRegister``.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: ``hipHostRegister`` flags. Defaults to ``0``
                (``hipHostRegisterDefault``). Pass ``0x02``
                (``hipHostRegisterMapped``) to additionally map the
                region into the device address space.

        Returns:
            True if ``hipHostRegister`` succeeded, False otherwise.
        """
        if self._libhip is None:
            return False
        try:
            err = self._libhip.hipHostRegister(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(flags),
            )
            return err == 0
        except Exception as exc:
            logger.warning(
                "hipHostRegister failed for ptr=%#x size=%d: %s",
                ptr,
                size,
                exc,
            )
            return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if ``hipHostUnregister`` succeeded, False otherwise.
        """
        if self._libhip is None:
            return False
        try:
            err = self._libhip.hipHostUnregister(ctypes.c_void_p(ptr))
            return err == 0
        except Exception as exc:
            logger.warning(
                "hipHostUnregister failed for ptr=%#x: %s", ptr, exc
            )
            return False

    @property
    def is_pin_supported(self) -> bool:
        """Whether ROCm memory pinning is supported.

        Returns:
            True if ``libamdhip64.so`` is available, False otherwise.
        """
        return self._libhip is not None
