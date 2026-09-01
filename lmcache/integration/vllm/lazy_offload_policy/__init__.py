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
    """FIFO is the legacy count-triggered drain; see the design doc."""

    FIFO = "FIFO"
    EVICTION_AWARE = "EVICTION_AWARE"


def create_offload_policy(
    configs: dict[str, ConfigValue],
    gpu_block_pool: "BlockPool",
) -> OffloadPolicy:
    """Build the policy ``POLICY_CONFIG_KEY`` names, EVICTION_AWARE default.

    The remaining config keys are read by the selected policy itself; the
    eviction-aware one ranks ``gpu_block_pool`` by eviction order.

    Raises:
        ValueError: If the name is unknown or a tunable is out of range.
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
