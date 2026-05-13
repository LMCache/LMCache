# SPDX-License-Identifier: Apache-2.0
"""Platform backend registry.

Each accelerator sub-package (``platform/cuda``, ``platform/cpu``,
future ``platform/xpu`` ...) registers a concrete factory for the
``ExternalStreamLike`` primitive consumed by the multiprocess server.

The :func:`make_external_stream` dispatcher in
:mod:`lmcache.v1.platform.stream` looks up the factory here based on
``lmcache.torch_device_type`` (and falls back to the ``"cpu"`` entry
when no concrete backend matches).  Adding a new accelerator therefore
requires *zero* changes to the dispatcher file; it only needs to ship
its own sub-package and register the right callable at import time.

Device-context activation and Event creation have moved off the
registry: callers go through ``lmcache.torch_dev`` directly, which is
itself dispatched per-platform by :mod:`lmcache.__init__._detect_device`.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"


# Per-capability table: ``{device_type: factory}``.
_STREAM_FACTORIES: Dict[str, Callable[..., Any]] = {}

# KV-cache IPC wrapper factory per device type. Concrete sub-packages
# self-register here (CUDA -> ``CudaIPCWrapper``, CPU -> POSIX-SHM
# wrapper) so :func:`get_kv_wrapper_factory` can dispatch by
# ``tensor.device.type`` without any if/elif chain in the call site.
_KV_WRAPPER_FACTORIES: Dict[str, Callable[..., Any]] = {}

# Per-backend availability predicate (e.g. ``torch.cuda.is_available``).
# The dispatcher consults this to skip a registered backend when its
# runtime requirements are not met (e.g. CUDA build present but no GPU
# actually available).  Missing entry == always available.
_AVAILABILITY: Dict[str, Callable[[], bool]] = {}


def register_availability(device_type: str, predicate: Callable[[], bool]) -> None:
    _AVAILABILITY[device_type] = predicate


def register_stream(device_type: str, factory: Callable[..., Any]) -> None:
    _STREAM_FACTORIES[device_type] = factory


def register_kv_wrapper(device_type: str, factory: Callable[..., Any]) -> None:
    """Register a KV-cache IPC wrapper factory for ``device_type``.

    The factory takes a single ``torch.Tensor`` and returns a wrapper
    instance ready to be sent over the multiprocess wire.
    """
    _KV_WRAPPER_FACTORIES[device_type] = factory


def is_available(device_type: str) -> bool:
    pred = _AVAILABILITY.get(device_type)
    if pred is None:
        return True
    try:
        return bool(pred())
    except Exception:
        return False


def get_stream_factory(device_type: str) -> Callable[..., Any] | None:
    """Pick the active stream factory for ``device_type`` with CPU fallback.

    The lookup honours the runtime availability predicate: a backend
    that registered itself but reports ``is_available() == False`` is
    skipped so the caller transparently falls through to the default
    backend.
    """
    factory = _STREAM_FACTORIES.get(device_type)
    if factory is not None and is_available(device_type):
        return factory
    return _STREAM_FACTORIES.get(DEFAULT_BACKEND)


def get_kv_wrapper_factory(device_type: str) -> Callable[..., Any]:
    """Pick the KV-cache wrapper factory for ``device_type``.

    Unlike :func:`get_stream_factory`, this lookup does NOT silently
    fall back: a missing entry means the caller is asking for a
    backend that nobody registered (typically because the relevant
    sub-package was not imported), which is a programming error and
    deserves an explicit failure.
    """
    factory = _KV_WRAPPER_FACTORIES.get(device_type)
    if factory is None:
        raise ValueError(
            "No KV-cache wrapper factory registered for device type %r" % device_type
        )
    return factory


def snapshot() -> Dict[str, Dict[str, Callable[..., Any]]]:
    """Return a deep-copy of the registry tables.

    Test suites use this to install backend overrides without leaking
    state across tests; pair with :func:`restore` in a ``finally`` /
    fixture teardown clause.
    """
    return {
        "stream": dict(_STREAM_FACTORIES),
        "kv_wrapper": dict(_KV_WRAPPER_FACTORIES),
        "availability": dict(_AVAILABILITY),
    }


def restore(state: Dict[str, Dict[str, Callable[..., Any]]]) -> None:
    """Restore registry tables to a previously :func:`snapshot`-ed state."""
    _STREAM_FACTORIES.clear()
    _STREAM_FACTORIES.update(state["stream"])
    _KV_WRAPPER_FACTORIES.clear()
    _KV_WRAPPER_FACTORIES.update(state.get("kv_wrapper", {}))
    _AVAILABILITY.clear()
    _AVAILABILITY.update(state["availability"])
