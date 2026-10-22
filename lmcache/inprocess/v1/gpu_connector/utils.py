# SPDX-License-Identifier: Apache-2.0
"""Helpers for the in-process engine's GPU connectors."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig) -> bool:
    """
    Check if the GPU Connector needs to create an intermediate
    buffer on the GPU
    """
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def assert_layerwise_gpu_connector(gpu_connector: "GPUConnectorInterface") -> None:
    """
    Assert that a GPU Connector is a layerwise connector.
    """
    # Import at runtime to avoid circular dependency
    # First Party
    from lmcache.inprocess.v1.gpu_connector import gpu_connectors, xpu_connectors

    valid_connectors = (
        gpu_connectors.VLLMPagedMemLayerwiseGPUConnector,
        gpu_connectors.VLLMBufferLayerwiseGPUConnector,
        xpu_connectors.VLLMPagedMemLayerwiseXPUConnector,
        xpu_connectors.VLLMBufferLayerwiseXPUConnector,
    )

    assert isinstance(gpu_connector, valid_connectors)
