# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives.

Importing this package self-registers concrete factories with
:mod:`lmcache.v1.platform._registry` so the cross-platform dispatchers
in :mod:`lmcache.v1.platform.stream` and
:mod:`lmcache.v1.platform.device_ctx` can locate them by device type.
"""

# Third Party
import torch

# First Party
from lmcache.v1.platform._registry import (
    register_availability,
    register_device_ctx,
    register_event,
    register_ipc_event,
    register_stream,
)
from lmcache.v1.platform.cuda.device_ctx import make_cuda_device_context


def _make_cuda_event(device):  # noqa: ARG001
    return torch.cuda.Event(interprocess=True)


def _from_cuda_ipc_handle(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)


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
register_device_ctx("cuda", make_cuda_device_context)
register_event("cuda", _make_cuda_event)
register_ipc_event("cuda", _from_cuda_ipc_handle)
