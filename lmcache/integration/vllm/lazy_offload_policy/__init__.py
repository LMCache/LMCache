# SPDX-License-Identifier: Apache-2.0
"""Lazy-offload policies used by the vLLM pending store."""

# Standard
from typing import TYPE_CHECKING, cast
import enum

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    ConfigValue,
    OffloadPolicy,
)
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    EvictionAwareStoreQueue,
    GPUBlockPoolView,
    LazyOffloadPolicyConfig,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)

#: Config key naming the policy to run.
POLICY_CONFIG_KEY = "lmcache.mp.lazy_offload_policy"


class LazyOffloadMode(enum.Enum):
    """Which drain policy buffers and releases store operations.

    Attributes:
        FIFO: Count-triggered whole-request drain; the legacy placeholder.
        EVICTION_AWARE: Pressure-triggered drain in free-queue LRU order.
    """

    FIFO = "FIFO"
    EVICTION_AWARE = "EVICTION_AWARE"


def create_offload_policy(
    configs: dict[str, ConfigValue],
    gpu_block_pool: "BlockPool",
) -> OffloadPolicy:
    """Build the configured lazy-offload policy.

    Args:
        configs: vLLM connector extra configuration. ``POLICY_CONFIG_KEY``
            selects the policy (``"EVICTION_AWARE"`` by default); the
            remaining keys are read by the selected policy itself.
        gpu_block_pool: The scheduler's GPU block pool, which the
            eviction-aware policy reads to rank blocks by eviction order.

    Returns:
        The policy instance the manager drives.

    Raises:
        ValueError: If the configured policy name is unknown, or a policy
            tunable is outside its documented range.
    """
    name = cast(
        str, configs.get(POLICY_CONFIG_KEY, LazyOffloadMode.EVICTION_AWARE.value)
    )
    try:
        mode = LazyOffloadMode(name)
    except ValueError as e:
        raise ValueError(f"Unknown offload policy: {name}") from e
    if mode is LazyOffloadMode.FIFO:
        return FIFOOffloadPolicy(configs)
    config = LazyOffloadPolicyConfig.from_configs(configs)
    logger.info("lazy offload enabled with EVICTION_AWARE policy: %s", config)
    return EvictionAwareStoreQueue(config, GPUBlockPoolView(gpu_block_pool))
