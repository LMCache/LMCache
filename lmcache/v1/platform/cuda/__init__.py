# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

Importing this package self-registers a CUDA KV-cache wrapper
factory with :mod:`lmcache.v1.platform._registry`, so the dispatch
in :mod:`lmcache.integration.vllm.vllm_multi_process_adapter` can
locate it by ``tensor.device.type``.
"""

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.platform._registry import (
    register_availability,
    register_kv_wrapper,
)


def _kv_wrapper_factory(tensor: torch.Tensor) -> Any:
    """Indirect-dispatch wrapper.

    Re-imports :class:`CudaIPCWrapper` on every call so test suites
    that swap the symbol still see their override take effect.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper

    return CudaIPCWrapper(tensor)


def _cuda_is_available() -> bool:
    """Lazy availability check to avoid circular import at module load."""
    # First Party
    from lmcache import torch_dev

    return torch_dev.is_available()


register_availability("cuda", _cuda_is_available)
register_kv_wrapper("cuda", _kv_wrapper_factory)
