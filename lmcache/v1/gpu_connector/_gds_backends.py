# SPDX-License-Identifier: Apache-2.0
"""Explicit construction of the built-in GDS backends.

To add an implementation, subclass GDSBackend and add its class to BACKENDS.
The interfaces and GDSContext do not need to change. Imports stay driver-free.
"""

# First Party
from lmcache.v1.gpu_connector._cufile_async import CuFileBackend
from lmcache.v1.gpu_connector._gds_async import GDSBackend
from lmcache.v1.gpu_connector._hipfile_async import HipFileBackend
from lmcache.v1.gpu_connector._phx_async import PhxBackend
from lmcache.v1.gpu_connector._ugds_async import UgdsBackend

BACKENDS: dict[str, type[GDSBackend]] = {
    backend.name: backend
    for backend in (CuFileBackend, HipFileBackend, UgdsBackend, PhxBackend)
}


def create_backend(name: str) -> GDSBackend:
    """Construct and validate a backend without loading its native driver.

    ``auto`` uses each implementation's default-selection rule. Unknown names
    and incompatible environments raise ValueError. No instance is cached.
    """
    if name == "auto":
        backend_class = next(
            (backend for backend in BACKENDS.values() if backend.is_default()), None
        )
        if backend_class is None:
            raise ValueError("no default GDS backend for this environment")
    else:
        backend_class = BACKENDS.get(name)
        if backend_class is None:
            raise ValueError(f"unsupported GDS L1 backend: {name}")
    backend = backend_class()
    backend.validate_environment()
    return backend
