# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives (fallbacks used when no
accelerator-backed implementation is available).

Importing this package self-registers the no-op CPU factories with
:mod:`lmcache.v1.platform._registry`, so dispatchers can pick them up
either as the active backend (on CPU-only hosts) or as the default
fallback when no concrete accelerator backend matches.
"""

# First Party
from lmcache.v1.platform._registry import (
    DEFAULT_BACKEND,
    register_device_ctx,
    register_event,
    register_ipc_event,
    register_stream,
)
from lmcache.v1.platform.cpu.device_ctx import (
    MockInterprocessEvent,
    NoopDeviceContext,
)
from lmcache.v1.platform.cpu.stream import MockExternalStream


def _make_cpu_stream(raw_ptr, device_index):  # noqa: ARG001
    return MockExternalStream(raw_ptr)


def _make_cpu_device_ctx(device, stream):  # noqa: ARG001
    return NoopDeviceContext()


def _make_cpu_event(device):  # noqa: ARG001
    return MockInterprocessEvent()


def _from_cpu_ipc_handle(device, handle):  # noqa: ARG001
    return MockInterprocessEvent()


register_stream(DEFAULT_BACKEND, _make_cpu_stream)
register_device_ctx(DEFAULT_BACKEND, _make_cpu_device_ctx)
register_event(DEFAULT_BACKEND, _make_cpu_event)
register_ipc_event(DEFAULT_BACKEND, _from_cpu_ipc_handle)
