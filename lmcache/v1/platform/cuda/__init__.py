# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

:class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper` carries
a ``device_type`` ClassVar and a ``wrap`` factory classmethod, which
:func:`~lmcache.v1.platform._registry._discover_wrappers_once` picks
up at run-time -- no static ``register_kv_wrapper`` needed.

The CUDA availability predicate is still registered statically here
so callers can check ``is_available("cuda")`` at import time.
"""

# First Party
from lmcache.v1.platform._registry import register_availability
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.device_ext import register_pin_memory_backend


def _cuda_is_available() -> bool:
    """Lazy availability check to avoid circular import at module load."""
    # First Party
    from lmcache import torch_dev

    return torch_dev.is_available()


register_availability("cuda", _cuda_is_available)
register_pin_memory_backend("cuda", CudaPinMemoryBackend)
