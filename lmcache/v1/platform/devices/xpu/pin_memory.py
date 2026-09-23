# SPDX-License-Identifier: Apache-2.0
"""XPU host-memory registration through the LMCache SYCL extension."""

# Standard
from typing import Protocol, cast
import importlib

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)


class _XpuOps(Protocol):
    """Native XPU operations required for host-memory registration."""

    def xpu_host_register(self, ptr: int, n_bytes: int) -> bool:
        """Register a host-memory range with the current XPU context."""
        ...

    def xpu_host_unregister(self, ptr: int) -> bool:
        """Release a host-memory range from the current XPU context."""
        ...


def _load_xpu_ops() -> _XpuOps | None:
    """Load and validate the native XPU host-registration operations.

    Returns:
        The native operation module when both registration operations are
        available, or ``None`` when the SYCL extension is not installed.
    """
    try:
        ops = importlib.import_module("lmcache.xpu_ops")
    except (ImportError, OSError, RuntimeError) as exc:
        logger.debug("XpuPinMemoryBackend: xpu_ops is unavailable: %s", exc)
        return None

    if not callable(getattr(ops, "xpu_host_register", None)) or not callable(
        getattr(ops, "xpu_host_unregister", None)
    ):
        logger.warning(
            "XpuPinMemoryBackend: xpu_ops lacks host-registration operations"
        )
        return None
    return cast(_XpuOps, ops)


class XpuPinMemoryBackend(PinMemoryBackend):
    """Register host memory for direct XPU DMA through ``lmcache.xpu_ops``.

    The native extension imports a host range into the current XPU SYCL
    context. This supports externally allocated memory, including shared
    ``mmap`` regions, that PyTorch's pinned-memory allocator cannot manage.
    """

    def __init__(self) -> None:
        """Discover the optional native XPU host-registration extension."""
        self._ops = _load_xpu_ops()

    @staticmethod
    def is_available() -> bool:
        """Return whether the native XPU registration backend can initialize.

        Returns:
            ``True`` when the native extension provides both required host
            registration operations; otherwise ``False``.
        """
        return _load_xpu_ops() is not None

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register one host-memory range with the active XPU context.

        Args:
            ptr: Raw base pointer of the host-memory range.
            size: Number of bytes to register.
            flags: Accepted for :class:`PinMemoryBackend` compatibility. SYCL
                host registration does not define registration flags.

        Returns:
            ``True`` when the native registration succeeds; otherwise
            ``False`` so callers can use their synchronous-copy fallback.
        """
        del flags
        ops = self._ops
        if ops is None or ptr == 0 or size <= 0:
            return False

        try:
            return ops.xpu_host_register(ptr, size)
        except RuntimeError as exc:
            logger.warning(
                "xpu_host_register failed for ptr=%#x size=%d: %s", ptr, size, exc
            )
            return False

    def unpin_memory(self, ptr: int) -> bool:
        """Release a host-memory range registered by :meth:`pin_memory`.

        Args:
            ptr: The original pointer passed to :meth:`pin_memory`.

        Returns:
            ``True`` when the native release succeeds; otherwise ``False``.
        """
        ops = self._ops
        if ops is None or ptr == 0:
            return False

        try:
            return ops.xpu_host_unregister(ptr)
        except RuntimeError as exc:
            logger.warning("xpu_host_unregister failed for ptr=%#x: %s", ptr, exc)
            return False

    @property
    def is_pin_supported(self) -> bool:
        """Return whether the native XPU host-registration extension exists."""
        return self._ops is not None
