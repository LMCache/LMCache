# SPDX-License-Identifier: Apache-2.0
"""Shared slab creation for filesystem-backed GDS implementations."""

# Standard
import os

# First Party
from lmcache.v1.gpu_connector._gds_async import GDSBackend, GDSHandle


class FileGDSBackend(GDSBackend):
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
