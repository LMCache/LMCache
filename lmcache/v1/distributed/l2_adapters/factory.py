# SPDX-License-Identifier: Apache-2.0
"""
Factory registry for L2 adapters.

Each adapter module self-registers a factory callable via
``register_l2_adapter_factory`` at import time.  The factory
signature is
``(L2AdapterConfigBase, Optional[L1MemoryDesc]) -> L2AdapterInterface``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
    )
    from lmcache.v1.distributed.l2_adapters.config import (
        L2AdapterConfigBase,
    )

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Type alias for factory callables:
#   (config, l1_memory_desc) -> L2AdapterInterface
L2AdapterFactory = Callable[
    ["L2AdapterConfigBase", "Optional[L1MemoryDesc]"],
    "L2AdapterInterface",
]

# -----------------------------------------------------------------
# Registry: adapter type name -> factory callable
# -----------------------------------------------------------------

_L2_ADAPTER_FACTORY_REGISTRY: dict[str, L2AdapterFactory] = {}


def register_l2_adapter_factory(
    name: str,
    factory: L2AdapterFactory,
) -> None:
    """Register an adapter factory for the given type
    name.

    Each adapter module should call this at import time
    **after** its config class has been registered via
    ``register_l2_adapter_type``.

    Args:
        name: Adapter type name (must match the name used
            in ``register_l2_adapter_type``).
        factory: ``(config, l1_memory_desc)``
            -> ``L2AdapterInterface``.
    """
    if name in _L2_ADAPTER_FACTORY_REGISTRY:
        raise ValueError("L2 adapter factory already registered: %s" % name)
    _L2_ADAPTER_FACTORY_REGISTRY[name] = factory


def create_l2_adapter_from_registry(
    config: "L2AdapterConfigBase",
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> "L2AdapterInterface":
    """Create an L2 adapter using the factory registry.

    Looks up the type name for *config* via the config
    registry, then calls the matching factory.

    Args:
        config: An adapter config instance.
        l1_memory_desc: Optional L1 memory descriptor,
            required by adapters that register L1 memory
            with an external backend (e.g. Nixl).

    Returns:
        A new ``L2AdapterInterface`` instance.

    Raises:
        ValueError: If no factory is registered for this
            config type.
    """
    # Import here to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import (
        get_type_name_for_config,
    )

    name = get_type_name_for_config(config)
    factory = _L2_ADAPTER_FACTORY_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            "No adapter factory registered for type "
            "%s. Make sure the adapter module is "
            "imported." % name
        )
    return factory(config, l1_memory_desc)
