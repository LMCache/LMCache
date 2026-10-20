# SPDX-License-Identifier: Apache-2.0
"""Common observable gauges for private and shared L1 bindings."""

# Standard
from collections.abc import Callable
import threading
import weakref

# First Party
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.mp_observability.otel_init import register_gauge

_lock = threading.Lock()
_registered = False
_targets: weakref.WeakValueDictionary[int, L1ManagerInterface] = (
    weakref.WeakValueDictionary()
)


def _usage_ratio(target: L1ManagerInterface) -> float:
    """Return the used/total ratio, or zero for an empty pool."""
    used, total = target.get_memory_usage()
    return used / total if total > 0 else 0.0


def _observations(
    value: Callable[[L1ManagerInterface], int | float],
) -> list[tuple[int | float, dict[str, object]]]:
    """Snapshot live bindings, then read their usage outside the registry lock."""
    with _lock:
        managers = list(_targets.values())
    return [
        (
            value(manager),
            {
                "l1_tag": manager.config.tag,
                "backend": "gds"
                if manager.config.gds_l1_config
                else "dram+devdax"
                if manager.config.memory_config.devdax_path
                and manager.config.memory_config.devdax_size_in_bytes
                else "devdax"
                if manager.config.memory_config.devdax_path
                else "dram",
            },
        )
        for manager in managers
    ]


def register_l1_metrics(manager: L1ManagerInterface) -> None:
    """Include a fully initialized binding in the process's L1 gauges.

    Args:
        manager: Binding to observe, with a unique process-local manager ID.
            Held weakly; call ``unregister_l1_metrics`` before discarding it.
    """
    global _registered
    with _lock:
        _targets[manager.l1_manager_id] = manager
        if _registered:
            return
        _registered = True
        register_gauge(
            "lmcache.l1_manager",
            "lmcache_mp.l1_memory_usage_bytes",
            "Bytes currently held in L1 cache",
            lambda: _observations(lambda m: m.get_memory_usage()[0]),
        )
        register_gauge(
            "lmcache.l1_manager",
            "lmcache_mp.l1_usage_ratio",
            "L1 used/total ratio (0.0–1.0)",
            lambda: _observations(_usage_ratio),
        )
        register_gauge(
            "lmcache.l1_manager",
            "lmcache_mp.l1_staging_bytes",
            "Bytes held by L1 staging objects (write-reserved, not admitted)",
            lambda: _observations(lambda m: m.get_staging_memory_usage()),
        )


def unregister_l1_metrics(manager: L1ManagerInterface) -> None:
    """Remove a closed binding from subsequent scrapes; safe to repeat.

    Args:
        manager: The binding previously registered with ``register_l1_metrics``.
    """
    with _lock:
        _targets.pop(manager.l1_manager_id, None)
