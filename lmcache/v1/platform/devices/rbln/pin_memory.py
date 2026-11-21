# SPDX-License-Identifier: Apache-2.0
"""torch-rbln host registration with device-scoped ownership."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)
_registration_scope = threading.local()


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
    """Own registrations by (address, device), serialized with runtime calls.

    The pointer-only unpin API releases every registration this backend owns
    at that address, even when the caller has switched devices. Failed releases
    remain owned and can be retried before the host allocation is destroyed.
    """

    def __init__(self) -> None:
        """Create an empty registration owner; no device is initialized."""
        self._registered: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register a host range on the current device without modifying its bytes.

        Args:
            ptr: Start address of the live host allocation.
            size: Registration length in bytes.
            flags: Ignored; retained for the platform interface.

        Returns:
            True after registration succeeds. False for an empty range, an
            unregistered allocation scope, a duplicate owned registration, or
            runtime failure.
        """
        del flags
        if getattr(_registration_scope, "skip", False):
            return False
        if ptr <= 0 or size <= 0:
            return False
        device = torch.rbln.current_device()
        key = (ptr, device)
        with self._lock:
            if key in self._registered:
                return False
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
            self._registered.add(key)
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
        with self._lock:
            keys = sorted(key for key in self._registered if key[0] == ptr)
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
