# SPDX-License-Identifier: Apache-2.0
"""ROCm host-memory registration through PyTorch's CUDA-compatible API."""

# Standard
from collections.abc import Callable
import ctypes
import ctypes.util
import functools

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.devices.cuda.pin_memory import CudaPinMemoryBackend

logger = init_logger(__name__)


class RocmPinMemoryBackend(CudaPinMemoryBackend):
    """ROCm memory pinning through PyTorch's CUDA-compatible runtime binding."""

    @staticmethod
    @functools.cache
    def _load_get_last_error() -> Callable[[], int] | None:
        """Bind ``hipGetLastError`` from the HIP runtime torch loaded.

        Returns:
            The bound ``hipGetLastError`` function, or ``None`` when the HIP
            runtime library cannot be loaded.
        """
        name = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        try:
            get_last_error = ctypes.CDLL(name).hipGetLastError
        except (AttributeError, OSError):
            logger.debug(
                "RocmPinMemoryBackend: cannot bind hipGetLastError from %s", name
            )
            return None
        get_last_error.restype = ctypes.c_int
        get_last_error.argtypes = []
        return get_last_error
