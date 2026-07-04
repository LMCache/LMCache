# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

:class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper` carries
a ``device_type`` ClassVar and a ``wrap`` factory classmethod, which
the universal registry picks up at run-time.

:class:`~lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend`
carries a ``device_type = "cuda"`` ClassVar, so the universal registry
discovers it automatically as well -- no ``register_pin_memory_backend``
call needed here.

The CUDA availability predicate is still registered statically here
so callers can check ``is_available("cuda")`` at import time.
"""

# First Party
from lmcache.v1.platform._registry import register_availability


def _cuda_is_available() -> bool:
    """Lazy availability check to avoid circular import at module load."""
    # First Party
    from lmcache import torch_dev

    return torch_dev.is_available()


register_availability("cuda", _cuda_is_available)
