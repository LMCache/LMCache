# SPDX-License-Identifier: Apache-2.0
"""Optional helpers for backends with registered POSIX descriptors."""

# Standard
from abc import abstractmethod
from typing import Any
import os

# First Party
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle


class FDGDSBackend(GDSBackend):
    """Share native descriptor registration with FDGDSHandle."""

    @abstractmethod
    def open_handle(self, fd: int, path: str) -> "FDGDSHandle":
        """Take ownership of fd; return a registered handle or close fd on failure."""

    @abstractmethod
    def register_handle(self, fd: int) -> Any:
        """Return a native registration, leaving fd ownership with the caller."""

    @abstractmethod
    def deregister_handle(self, handle: Any) -> None:
        """Release a native registration without closing its descriptor."""


class FileGDSBackend(FDGDSBackend):
    """Create and preallocate a regular file before registering it for IO."""

    def open_slab(self, location: str, size: int, direct_io: bool) -> GDSHandle:
        """Create location/lmcache_gds_slab.bin and return its owning handle.

        The file is truncated on initialization. Allocation and registration
        failures propagate after closing any descriptor opened by this method.
        """
        os.makedirs(location, exist_ok=True)
        path = os.path.join(location, "lmcache_gds_slab.bin")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.posix_fallocate(fd, 0, size)
        finally:
            os.close(fd)
        flags = os.O_RDWR
        if direct_io:
            flags |= os.O_DIRECT
        return self.open_handle(os.open(path, flags), path)


class FDGDSHandle(GDSHandle):
    """Own a POSIX descriptor and its native registration."""

    def __init__(self, backend: FDGDSBackend, fd: int, handle: Any, path: str) -> None:
        super().__init__(path)
        self._backend = backend
        self._fd = fd
        self._handle = handle

    @property
    def fd(self) -> int:
        """Return the owned descriptor, or -1 after close()."""
        return self._fd

    def close(self) -> None:
        """Deregister and close fd once, even if deregistration fails."""
        if self._fd < 0:
            return
        try:
            self._backend.deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1
