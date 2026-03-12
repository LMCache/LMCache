# SPDX-License-Identifier: Apache-2.0
"""
L2 adapter factory with automatic module discovery.

All adapter modules under this package that call
``register_l2_adapter_type`` and ``register_l2_adapter_factory``
at import time are discovered automatically.  To add a new
adapter, simply create a new module in this directory -- **no
changes to any existing file are needed**.
"""

# Standard
import importlib
import pkgutil

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)

# Auto-discover and import every module in this package so
# that each adapter's self-registration code runs.
_PACKAGE_PATH = __path__  # type: ignore[name-defined]
_PACKAGE_NAME = __name__
for _finder, _mod_name, _is_pkg in pkgutil.iter_modules(_PACKAGE_PATH):
    importlib.import_module(f".{_mod_name}", _PACKAGE_NAME)


def create_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create an L2 adapter from its config via the
    factory registry.

    Args:
        config: The adapter-specific config object.
        l1_memory_desc: Descriptor of the L1 memory buffer,
            required for adapters that register L1 memory
            with an external backend (e.g. Nixl).

    Returns:
        L2AdapterInterface: A new adapter instance.

    Raises:
        ValueError: If no factory is registered for
            the config type.
    """
    return create_l2_adapter_from_registry(
        config,
        l1_memory_desc=l1_memory_desc,
    )
