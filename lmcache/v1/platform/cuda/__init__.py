# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

Importing this package self-registers a :class:`CudaPlatform` instance
with :mod:`lmcache.v1.platform._registry` so the cross-platform
dispatcher in :mod:`lmcache.v1.platform.stream` can locate it by
device type.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.v1.platform._registry import Platform, register_platform


class CudaPlatform(Platform):
    """CUDA accelerator platform.

    Stream construction is delegated through an indirect re-import so
    test suites that ``mock.patch`` :func:`make_cuda_external_stream`
    at the module level still see their override take effect (the
    registry keeps a stable :class:`Platform` instance while the actual
    implementation can be swapped at runtime).
    """

    device_type = "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def make_external_stream(self, raw_ptr: int, device_index: int) -> Optional[Any]:
        # First Party
        from lmcache.v1.platform.cuda.stream import make_cuda_external_stream

        return make_cuda_external_stream(raw_ptr, device_index)


register_platform(CudaPlatform())
