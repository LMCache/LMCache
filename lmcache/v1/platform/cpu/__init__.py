# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives (fallbacks used when no
accelerator-backed implementation is available).

Importing this package self-registers the no-op CPU stream factory
with :mod:`lmcache.v1.platform._registry`, so the stream dispatcher
can pick it up either as the active backend (on CPU-only hosts) or
as the default fallback when no concrete accelerator backend matches.
"""

# First Party
from lmcache.v1.platform._registry import DEFAULT_BACKEND, register_stream
from lmcache.v1.platform.cpu.stream import MockExternalStream


def _make_cpu_stream(raw_ptr, device_index):  # noqa: ARG001
    return MockExternalStream(raw_ptr)


register_stream(DEFAULT_BACKEND, _make_cpu_stream)
