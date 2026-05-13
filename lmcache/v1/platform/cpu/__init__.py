# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives (fallbacks used when no
accelerator-backed implementation is available).

Importing this package self-registers the no-op CPU stream factory
with :mod:`lmcache.v1.platform._registry`, so the stream dispatcher
can pick it up either as the active backend (on CPU-only hosts) or
as the default fallback when no concrete accelerator backend matches.

Device-context and Event creation are no longer routed through the
registry: callers go through ``lmcache.torch_dev`` directly which on
CPU-only hosts is duck-typed by
:mod:`lmcache.v1.platform.cpu.torch_dev_stub`.
"""

# First Party
from lmcache.v1.platform._registry import (
    DEFAULT_BACKEND,
    register_kv_wrapper,
    register_stream,
)
from lmcache.v1.platform.cpu.stream import MockExternalStream


def _make_cpu_stream(raw_ptr, device_index):  # noqa: ARG001
    return MockExternalStream(raw_ptr)


register_stream(DEFAULT_BACKEND, _make_cpu_stream)


def _kv_wrapper_factory(tensor):
    """Indirect-dispatch wrapper, mirrors :func:`cuda._stream_factory`.

    Defers loading :mod:`lmcache.v1.platform.cpu.shm` (which pulls in
    ``multiprocess.custom_types``, which transitively reads
    ``lmcache.torch_dev``) until first use, so importing this package
    during ``lmcache/__init__.py``'s bootstrap does not race the
    ``torch_dev`` attach.
    """
    # First Party
    from lmcache.v1.platform.cpu.shm import migrate_to_shm_and_wrap

    return migrate_to_shm_and_wrap(tensor)


register_kv_wrapper("cpu", _kv_wrapper_factory)
