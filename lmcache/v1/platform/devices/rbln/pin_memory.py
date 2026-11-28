# SPDX-License-Identifier: Apache-2.0
"""torch-rbln host registration with device-scoped ownership."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import importlib.metadata
import threading

# Third Party
from packaging.version import Version
import torch

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)
_registration_scope = threading.local()
# First torch-rbln release with host_register and host_unregister.
_MIN_TORCH_RBLN = Version("0.12.1a0")


def _require_host_registration() -> None:
    """Fail with an upgrade message when torch-rbln predates host registration.

    Raises:
        RuntimeError: If the installed torch-rbln is older than 0.12.1a0.
    """
    installed = importlib.metadata.version("torch-rbln")
    if Version(installed) < _MIN_TORCH_RBLN:
        raise RuntimeError(
            f"Registering the SHM pool on RBLN requires torch-rbln "
            f"{_MIN_TORCH_RBLN} or later, but {installed} is installed; "
            "upgrade torch-rbln."
        )


@contextmanager
def skip_host_registration() -> Iterator[None]:
    """Decline registration on the thread allocating the server's SHM pool."""
    previous = getattr(_registration_scope, "skip", False)
    _registration_scope.skip = True
    try:
        yield
    finally:
        _registration_scope.skip = previous


class RblnPinMemoryBackend(PinMemoryBackend):
    """Own registrations by (address, device).

    The runtime serializes registration and refuses a range that overlaps one
    already registered on the device, so this backend takes no lock; it only
    records the device each registration belongs to. The pointer-only unpin
    API releases every registration this backend owns at that address, even
    when the caller has switched devices. Failed releases remain owned and can
    be retried before the host allocation is destroyed. Callers must not pin
    and unpin the same address concurrently.
    """

    def __init__(self) -> None:
        """Create an empty registration owner; no device is initialized."""
        self._registered: set[tuple[int, int]] = set()

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register a host range on the current device without modifying its bytes.

        Args:
            ptr: Start address of the live host allocation.
            size: Registration length in bytes.
            flags: Ignored; retained for the platform interface.

        Returns:
            True after registration succeeds. False for an empty range, inside
            ``skip_host_registration()``, or when the runtime refuses the range
            (for example, one already registered on this device) or cannot
            register host memory.

        Raises:
            RuntimeError: If the installed torch-rbln is older than 0.12.1a0.
        """
        del flags
        if getattr(_registration_scope, "skip", False):
            return False
        if ptr <= 0 or size <= 0:
            return False
        _require_host_registration()
        device = torch.rbln.current_device()
        try:
            registered = torch.rbln.host_register(ptr, size, device=device)
        except RuntimeError:
            logger.exception(
                "host_register(ptr=%#x, size=%d, device=%d) failed",
                ptr,
                size,
                device,
            )
            return False
        if not registered:
            return False
        self._registered.add((ptr, device))
        return True

    def unpin_memory(self, ptr: int) -> bool:
        """Release all owned registrations at an address on their original devices.

        The runtime drains pending transfers before releasing each registration.
        The host allocation must remain alive until this method succeeds.

        Args:
            ptr: Start address previously registered through this backend.

        Returns:
            True if all registrations were released. False if none was owned or
            any release failed; a retry releases only registrations still owned.
        """
        # Iterate a copy: another thread may register a different address
        # meanwhile, and a set must not change size during iteration.
        keys = sorted(key for key in self._registered.copy() if key[0] == ptr)
        if not keys:
            return False
        for key in keys:
            _, device = key
            try:
                torch.rbln.host_unregister(ptr, device=device)
            except RuntimeError:
                logger.exception(
                    "host_unregister(ptr=%#x, device=%d) failed; keeping ownership",
                    ptr,
                    device,
                )
                return False
            self._registered.remove(key)
        return True

    @property
    def is_pin_supported(self) -> bool:
        """Report host registration as supported (torch-rbln 0.12.1a0+)."""
        return True
