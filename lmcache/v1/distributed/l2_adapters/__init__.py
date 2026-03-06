# SPDX-License-Identifier: Apache-2.0
"""
L2 adapter factory.

Provides ``create_l2_adapter`` to instantiate an L2 adapter from its config.
"""

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    MockL2AdapterConfig,
    NixlStoreL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2Adapter


def create_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """
    Create an L2 adapter instance from its config.

    The concrete config type determines which adapter class to instantiate.

    Args:
        config: The adapter-specific config object.
        l1_memory_desc: Descriptor of the L1 memory buffer, required for adapters
            that register L1 memory with an external backend (e.g. Nixl).

    Returns:
        L2AdapterInterface: A new adapter instance.

    Raises:
        ValueError: If the config type is not recognized, or if a required
            argument (e.g. l1_memory_desc) is missing for the given config type.
    """
    if isinstance(config, MockL2AdapterConfig):
        return MockL2Adapter(config)

    if isinstance(config, NixlStoreL2AdapterConfig):
        # Lazy import nixl
        # First Party
        from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (
            NixlStoreL2Adapter,
        )

        if l1_memory_desc is None:
            raise ValueError(
                "l1_memory_desc is required to create a NixlStoreL2Adapter."
            )
        return NixlStoreL2Adapter(config, l1_memory_desc)

    raise ValueError(
        f"Unknown L2 adapter config type: {type(config).__name__}. "
        f"Add a branch in create_l2_adapter() for this config type."
    )
