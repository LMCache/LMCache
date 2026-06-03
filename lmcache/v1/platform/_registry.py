# SPDX-License-Identifier: Apache-2.0
"""Platform backend registry.

Each accelerator sub-package (``platform/cuda``, ``platform/cpu``,
future ``platform/xpu`` ...) registers a concrete factory for the
KV-cache IPC wrapper consumed by the multiprocess adapter.

The :func:`get_kv_wrapper_factory` lookup keys on
``tensor.device.type`` so the call site in
:mod:`lmcache.integration.vllm.vllm_multi_process_adapter` stays free
of any if/elif chain. Adding a new accelerator therefore requires
*zero* changes to the dispatcher; it only needs to ship its own
sub-package and register the right callable at import time.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"


# KV-cache IPC wrapper factory per device type. Concrete sub-packages
# self-register here (CUDA -> ``CudaIPCWrapper``, CPU -> POSIX-SHM
# wrapper) so :func:`get_kv_wrapper_factory` can dispatch by
# ``tensor.device.type`` without any if/elif chain in the call site.
_KV_WRAPPER_FACTORIES: Dict[str, Callable[..., Any]] = {}

# Per-backend availability predicate (e.g. CUDA's ``is_available``).
# Missing entry == always available.
_AVAILABILITY: Dict[str, Callable[[], bool]] = {}


def register_availability(device_type: str, predicate: Callable[[], bool]) -> None:
    """Register an availability predicate for a device type.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).
        predicate: A zero-argument callable returning ``True`` when the
            device is available.
    """
    _AVAILABILITY[device_type] = predicate


def register_kv_wrapper(device_type: str, factory: Callable[..., Any]) -> None:
    """Register a KV-cache IPC wrapper factory for ``device_type``.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).
        factory: A callable that takes a single ``torch.Tensor`` and
            returns a wrapper instance ready for the multiprocess wire.
    """
    _KV_WRAPPER_FACTORIES[device_type] = factory


def is_available(device_type: str) -> bool:
    """Check whether a device type is available.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        ``True`` if the device is available or no predicate is registered,
        ``False`` otherwise.
    """
    pred = _AVAILABILITY.get(device_type)
    if pred is None:
        return True
    try:
        return bool(pred())
    except Exception:
        return False


def get_kv_wrapper_factory(device_type: str) -> Callable[..., Any]:
    """Pick the KV-cache wrapper factory for ``device_type``.

    A missing entry means the caller is asking for a backend that
    nobody registered (typically because the relevant sub-package was
    not imported), which is a programming error and deserves an
    explicit failure.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        The registered KV-cache wrapper factory for the device type.

    Raises:
        ValueError: If no factory is registered for the device type.
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

    Returns:
        A dict with keys ``"kv_wrapper"`` and ``"availability"``, each
        mapping device-type strings to their registered callables.
    """
    return {
        "kv_wrapper": dict(_KV_WRAPPER_FACTORIES),
        "availability": dict(_AVAILABILITY),
    }


def restore(state: Dict[str, Dict[str, Callable[..., Any]]]) -> None:
    """Restore registry tables to a previously :func:`snapshot`-ed state.

    Args:
        state: A snapshot dict as returned by :func:`snapshot`.
    """
    _KV_WRAPPER_FACTORIES.clear()
    _KV_WRAPPER_FACTORIES.update(state.get("kv_wrapper", {}))
    _AVAILABILITY.clear()
    _AVAILABILITY.update(state.get("availability", {}))
