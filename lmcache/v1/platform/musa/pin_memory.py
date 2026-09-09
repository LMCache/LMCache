# SPDX-License-Identifier: Apache-2.0
"""MUSA host-memory registration through the TorchMUSA runtime binding."""

# Standard
from typing import Protocol, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)


class _MusaRuntime(Protocol):
    """TorchMUSA runtime surface required for host-memory registration."""

    def musaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        """Register one host-memory range."""
        ...

    def musaHostUnregister(self, ptr: int) -> int:
        """Unregister one host-memory range."""
        ...


class MusaPinMemoryBackend(PinMemoryBackend):
    """Host-memory pinning backed by ``torch.musa.musart()``.

    The backend only probes for the two required runtime entry points during
    construction. It does not query device availability or create a MUSA
    context, so configuration parsing remains lightweight.
    """

    def __init__(self) -> None:
        """Discover the TorchMUSA host register and unregister functions.

        Missing or incomplete runtime bindings leave the backend unsupported;
        discovery never raises an exception to the platform capability gate.
        """
        self._musart: _MusaRuntime | None = None

        try:
            musa = getattr(torch, "musa", None)
            if musa is None:
                raise AttributeError("torch.musa is unavailable")
            musart = musa.musart()
            if not callable(getattr(musart, "musaHostRegister", None)):
                raise AttributeError("musaHostRegister is unavailable")
            if not callable(getattr(musart, "musaHostUnregister", None)):
                raise AttributeError("musaHostUnregister is unavailable")
        except Exception as exc:
            logger.debug(
                "MusaPinMemoryBackend: TorchMUSA host registration unavailable: %s",
                exc,
            )
            return

        self._musart = cast(_MusaRuntime, musart)

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register a host memory region with the MUSA runtime.

        Args:
            ptr: Raw base pointer of the host memory region.
            size: Number of bytes to register.
            flags: ``musaHostRegister`` flags. Lazy allocation passes ``0x02``
                (``musaHostRegisterMapped``).

        Returns:
            ``True`` only when ``musaHostRegister`` returns status zero.
        """
        musart = self._musart
        if musart is None:
            return False

        try:
            return int(musart.musaHostRegister(ptr, size, flags)) == 0
        except Exception as exc:
            logger.warning(
                "musaHostRegister failed for ptr=%#x size=%d flags=%#x: %s",
                ptr,
                size,
                flags,
                exc,
            )
            return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unregister a host memory region from the MUSA runtime.

        Args:
            ptr: Raw base pointer originally passed to :meth:`pin_memory`.

        Returns:
            ``True`` only when ``musaHostUnregister`` returns status zero.
        """
        musart = self._musart
        if musart is None:
            return False

        try:
            result = int(musart.musaHostUnregister(ptr))
        except Exception as exc:
            logger.warning("musaHostUnregister failed for ptr=%#x: %s", ptr, exc)
            return False

        if result != 0:
            logger.warning(
                "musaHostUnregister returned error code %d for ptr=%#x",
                result,
                ptr,
            )
            return False
        return True

    @property
    def is_pin_supported(self) -> bool:
        """Return whether both required TorchMUSA runtime functions exist.

        Returns:
            ``True`` when host registration and unregistration are callable.
        """
        return self._musart is not None
