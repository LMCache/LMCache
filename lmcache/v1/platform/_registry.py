# SPDX-License-Identifier: Apache-2.0
"""Platform backend registry.

Each accelerator sub-package (``platform/cuda``, ``platform/cpu``,
future ``platform/xpu`` ...) ships a concrete
:class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
subclass with a ``device_type`` ClassVar and a ``wrap`` factory
classmethod.  :func:`_discover_wrappers_once` scans the ``platform``
package for those subclasses at run-time and populates the factory
table -- no static ``register_kv_wrapper`` calls needed.

The :func:`get_kv_wrapper_factory` lookup keys on
``tensor.device.type`` so the call site in
:mod:`lmcache.integration.vllm.vllm_multi_process_adapter` stays free
of any if/elif chain.  Adding a new accelerator therefore requires
*zero* changes to the dispatcher; it only needs to ship its own
sub-package with a ``DeviceIPCWrapper`` subclass that sets
``device_type`` and ``wrap``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"


# KV-cache IPC wrapper factory per device type.  Populated lazily on
# first :func:`get_kv_wrapper_factory` call by scanning the
# ``platform`` package for
# :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
# subclasses.  Tests substitute entries via
# :func:`snapshot` / :func:`restore`.
_KV_WRAPPER_FACTORIES: Dict[str, Callable[..., Any]] = {}

# Per-backend availability predicate (e.g. CUDA's ``is_available``).
# Missing entry == always available.
_AVAILABILITY: Dict[str, Callable[[], bool]] = {}

# Guard so discovery only runs once (lazy init).  The lock plus the
# double-checked flag below keep the first concurrent caller from
# racing a second one through the scan and emitting duplicate
# "multiple wrappers claim device_type=..." warnings.
_WRAPPERS_DISCOVERED: bool = False
_DISCOVERY_LOCK = threading.Lock()


def _discover_wrappers_once() -> None:
    """Populate :data:`_KV_WRAPPER_FACTORIES` on first use.

    Walks ``lmcache.v1.platform`` two levels deep for
    :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
    subclasses.  Each subclass is indexed by its *device_type*
    ClassVar, and its *wrap* factory is stored as the KV-wrapper
    factory — but only when ``_is_default_wrapper`` is ``True``
    (so e.g. :class:`~lmcache.v1.platform.cuda.ipc_wrapper.RawCudaIPCWrapper`
    is skipped in favour of
    :class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper`).

    Subclasses with an empty *device_type* or ``_is_default_wrapper ==
    False`` are skipped.  Multiple subclasses claiming the same
    *device_type* trigger a warning; the first one wins.
    """
    global _WRAPPERS_DISCOVERED
    # Fast path: avoid the lock once discovery is done (the common case).
    if _WRAPPERS_DISCOVERED:
        return

    with _DISCOVERY_LOCK:
        # Re-check under the lock: another thread may have run the
        # scan while we were waiting.
        if _WRAPPERS_DISCOVERED:
            return

        # First Party
        from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper
        from lmcache.v1.utils.subclass_discovery import discover_subclasses
        import lmcache.v1.platform as platform_pkg

        for cls in discover_subclasses(
            platform_pkg,
            DeviceIPCWrapper,  # type: ignore[type-abstract]
            levels=[2, 2],
        ):
            _register_discovered_wrapper(cls)

        _WRAPPERS_DISCOVERED = True


def _register_discovered_wrapper(cls: type) -> None:
    """Index *cls* in :data:`_KV_WRAPPER_FACTORIES` by its device_type.

    Only registers when ``_is_default_wrapper`` is ``True`` so sibling
    subclasses (e.g. ``RawCudaIPCWrapper`` vs ``CudaIPCWrapper``) can
    share a ``device_type`` without colliding.
    """
    if not getattr(cls, "_is_default_wrapper", False):
        return

    device_type: str = getattr(cls, "device_type", "")
    if not device_type:
        logger.warning(
            "Skipping %s: empty device_type ClassVar; concrete "
            "DeviceIPCWrapper subclasses must override it.",
            cls.__name__,
        )
        return

    factory = getattr(cls, "wrap", cls)
    existing = _KV_WRAPPER_FACTORIES.get(device_type)
    if existing is not None and existing is not factory:
        logger.warning(
            "Multiple KV-wrapper classes claim device_type=%r "
            "(%s vs %s); keeping the first.",
            device_type,
            getattr(existing, "__name__", str(existing)),
            cls.__name__,
        )
        return

    _KV_WRAPPER_FACTORIES[device_type] = factory


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

    This is the manual registration path kept for backward
    compatibility.  New backends should instead set ``device_type``
    and ``wrap`` on their :class:`DeviceIPCWrapper` subclass and let
    :func:`_discover_wrappers_once` handle registration.

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

    Triggers lazy auto-discovery on first call (see
    :func:`_discover_wrappers_once`).  A missing entry means no
    :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
    subclass declared *device_type* for the requested backend.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        The registered KV-cache wrapper factory for the device type.

    Raises:
        ValueError: If no factory is registered for the device type.
    """
    _discover_wrappers_once()
    factory = _KV_WRAPPER_FACTORIES.get(device_type)
    if factory is None:
        raise ValueError(
            "No KV-cache wrapper factory registered for device type %r" % device_type
        )
    return factory


def snapshot() -> Dict[str, Any]:
    """Return a deep-copy of the registry tables.

    Test suites use this to install backend overrides without leaking
    state across tests; pair with :func:`restore` in a ``finally`` /
    fixture teardown clause.

    The lazy-discovery flag is captured alongside the tables: if a test
    snapshots *before* discovery runs and restores *after*, the next
    caller still re-runs discovery and picks up the auto-registered
    backends, instead of seeing a stale "already discovered, table is
    empty" view.

    Returns:
        A dict with keys ``"kv_wrapper"``, ``"availability"`` and
        ``"discovered"``.
    """
    return {
        "kv_wrapper": dict(_KV_WRAPPER_FACTORIES),
        "availability": dict(_AVAILABILITY),
        "discovered": _WRAPPERS_DISCOVERED,
    }


def restore(state: Dict[str, Any]) -> None:
    """Restore registry tables to a previously :func:`snapshot`-ed state.

    Args:
        state: A snapshot dict as returned by :func:`snapshot`.
    """
    global _WRAPPERS_DISCOVERED
    _KV_WRAPPER_FACTORIES.clear()
    _KV_WRAPPER_FACTORIES.update(state.get("kv_wrapper", {}))
    _AVAILABILITY.clear()
    _AVAILABILITY.update(state.get("availability", {}))
    _WRAPPERS_DISCOVERED = bool(state.get("discovered", False))


def reset_for_tests() -> None:
    """Wipe registry tables and force re-discovery on next access.

    Intended **only** for test fixtures: clears every registered KV
    wrapper / availability predicate and flips
    :data:`_WRAPPERS_DISCOVERED` back to ``False`` so the next
    :func:`get_kv_wrapper_factory` call re-runs the
    :func:`_discover_wrappers_once` scan and re-populates the table
    from the live ``platform`` sub-packages.

    This is the recommended replacement for callers that previously
    hand-mutated module-private globals; pair with an ``autouse``
    pytest fixture to guarantee every test starts and ends with a
    clean slate (see ``tests/v1/multiprocess/conftest.py``).
    """
    global _WRAPPERS_DISCOVERED
    _KV_WRAPPER_FACTORIES.clear()
    _AVAILABILITY.clear()
    _WRAPPERS_DISCOVERED = False
