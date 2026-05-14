# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

Importing this package self-registers the CUDA stream factory with
:mod:`lmcache.v1.platform._registry` so the cross-platform dispatcher
in :mod:`lmcache.v1.platform.stream` can locate it by device type.
"""

# Third Party
import torch

# First Party
from lmcache.v1.platform._registry import (
    register_availability,
    register_stream,
)


def _stream_factory(raw_ptr, device_index):
    """Indirect-dispatch wrapper.

    Re-imports :func:`make_cuda_external_stream` on every call so test
    suites that ``mock.patch`` the symbol at the module level still see
    their override take effect (the registry keeps a stable callable
    while the actual implementation can be swapped at runtime).
    """
    # First Party
    from lmcache.v1.platform.cuda.stream import make_cuda_external_stream

    return make_cuda_external_stream(raw_ptr, device_index)


register_availability("cuda", lambda: torch.cuda.is_available())
register_stream("cuda", _stream_factory)
