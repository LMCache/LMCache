# SPDX-License-Identifier: Apache-2.0
"""GCU host-memory pinning: try ``torch.gcu.gcurt()``, then ``libtopsrt``.

Mirrors :class:`~lmcache.v1.platform.devices.cuda.pin_memory.CudaPinMemoryBackend`
but backs the pin/unpin operations with the GCU runtime
(``topsHostRegister`` / ``topsHostUnregister``). Pinning is required for
asynchronous D2H/H2D DMA; when it is unavailable callers degrade to
synchronous copies.
"""

# Future
from __future__ import annotations

# Standard
import ctypes

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)


def _load_libtopsrt() -> ctypes.CDLL | None:
    """Load ``libtopsrt.so`` and bind the GCU host-register symbols.

    ``import torch_gcu`` first so the shared library is on the loader path; the
    symbols are bound with the ``topsHostRegister(void*, size_t, unsigned int)``
    and ``topsHostUnregister(void*)`` signatures from ``tops_runtime_api.h``.

    Returns:
        The bound ``ctypes.CDLL`` on success, or ``None`` when unavailable.
    """
    try:
        # Third Party
        import torch_gcu  # noqa: F401  # ensures libtopsrt is on the loader path
    except ImportError:
        pass
    try:
        lib = ctypes.CDLL("libtopsrt.so")
    except OSError as exc:
        logger.debug("GcuPinMemoryBackend: failed to load libtopsrt.so: %s", exc)
        return None
    lib.topsHostRegister.restype = ctypes.c_int
    lib.topsHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.topsHostUnregister.restype = ctypes.c_int
    lib.topsHostUnregister.argtypes = [ctypes.c_void_p]
    return lib


class GcuPinMemoryBackend(PinMemoryBackend):
    """GCU memory pinning backend.

    Pinning prefers ``torch.gcu.gcurt().topsHostRegister`` (the runtime already
    loaded by ``torch_gcu``). When that is unavailable the backend falls back to
    loading ``libtopsrt.so`` directly via :mod:`ctypes`.

    Attributes:
        _gcurt: The ``torch.gcu.gcurt()`` runtime handle, or ``None``.
        _libtopsrt: The ``ctypes``-loaded GCU runtime, or ``None``.
    """

    def __init__(self) -> None:
        """Discover a runtime binding without raising on GCU-less hosts."""
        self._gcurt = None
        self._libtopsrt = None

        try:
            # Third Party
            import torch

            gcu = getattr(torch, "gcu", None)
            if gcu is not None and gcu.is_available():
                gcurt_fn = getattr(gcu, "gcurt", None)
                if callable(gcurt_fn):
                    self._gcurt = gcurt_fn()
        except Exception as exc:
            logger.debug("GcuPinMemoryBackend: torch.gcu.gcurt() unavailable: %s", exc)

        if self._gcurt is None:
            self._libtopsrt = _load_libtopsrt()

        if self._gcurt is None and self._libtopsrt is None:
            logger.warning(
                "GcuPinMemoryBackend: neither torch.gcu.gcurt() nor libtopsrt "
                "is available; GCU host pinning disabled, D2H/H2D will be "
                "synchronous"
            )

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region using ``topsHostRegister``.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: ``topsHostRegister`` flags (default 0).

        Returns:
            True if registration succeeded, False otherwise.
        """
        # GCU requires Portable (0x01) for multi-GPU DMA: topsHostRegister
        # registers host memory only against the current device unless
        # Portable is set, so a cross-device D2H/H2D otherwise fails with
        # "not REGISTER_PORTABLE for device[N]". OR it in here so every GCU
        # pin is cross-device visible regardless of the caller's flags.
        flags |= 0x01  # cudaHostRegisterPortable
        if self._gcurt is not None:
            try:
                return int(self._gcurt.topsHostRegister(ptr, size, flags)) == 0
            except Exception as exc:
                logger.warning(
                    "topsHostRegister failed for ptr=%#x size=%d: %s", ptr, size, exc
                )
                return False
        if self._libtopsrt is not None:
            try:
                return (
                    self._libtopsrt.topsHostRegister(
                        ctypes.c_void_p(ptr),
                        ctypes.c_size_t(size),
                        ctypes.c_uint(flags),
                    )
                    == 0
                )
            except Exception as exc:
                logger.warning(
                    "topsHostRegister (ctypes) failed for ptr=%#x size=%d: %s",
                    ptr,
                    size,
                    exc,
                )
                return False
        return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if ``topsHostUnregister`` succeeded, False otherwise.
        """
        if self._gcurt is not None:
            try:
                return int(self._gcurt.topsHostUnregister(ptr)) == 0
            except Exception as exc:
                logger.warning("topsHostUnregister failed for ptr=%#x: %s", ptr, exc)
                return False
        if self._libtopsrt is not None:
            try:
                return self._libtopsrt.topsHostUnregister(ctypes.c_void_p(ptr)) == 0
            except Exception as exc:
                logger.warning(
                    "topsHostUnregister (ctypes) failed for ptr=%#x: %s", ptr, exc
                )
                return False
        return False

    @property
    def is_pin_supported(self) -> bool:
        """Whether GCU host-memory pinning is available."""
        return self._gcurt is not None or self._libtopsrt is not None
