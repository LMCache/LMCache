# SPDX-License-Identifier: Apache-2.0
"""
L2 adapter factory.

Provides ``create_l2_adapter`` to instantiate an L2 adapter from its config.
"""

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2Adapter


def create_l2_adapter(config: L2AdapterConfigBase) -> L2AdapterInterface:
    """
    Create an L2 adapter instance from its config.

    The concrete config type determines which adapter class to instantiate.

    Args:
        config: The adapter-specific config object.

    Returns:
        L2AdapterInterface: A new adapter instance.

    Raises:
        ValueError: If the config type is not recognized.
    """
    if isinstance(config, MockL2AdapterConfig):
        return MockL2Adapter(config)

    raise ValueError(
        f"Unknown L2 adapter config type: {type(config).__name__}. "
        f"Add a branch in create_l2_adapter() for this config type."
    )
