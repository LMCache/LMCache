# SPDX-License-Identifier: Apache-2.0
"""LMCache multi-process integration for NVIDIA TensorRT-LLM.

Start the standalone LMCache server, then select the ``lmcache-mp`` preset::

    kv_connector_config:
      connector: lmcache-mp
      server_url: grpc://localhost:5555

The scheduler and worker can also be configured explicitly::

    kv_connector_config:
      connector_module: lmcache.integration.tensorrt_llm.tensorrt_mp_adapter
      connector_scheduler_class: LMCacheMPKvConnectorScheduler
      connector_worker_class: LMCacheMPKvConnectorWorker
"""

# TensorRT-LLM is optional for core LMCache imports and documentation builds.
__all__: list[str] = []

try:
    # First Party
    from lmcache.integration.tensorrt_llm.tensorrt_mp_adapter import (
        LMCacheMPKvConnectorScheduler,
        LMCacheMPKvConnectorWorker,
    )

    __all__ = [
        "LMCacheMPKvConnectorScheduler",
        "LMCacheMPKvConnectorWorker",
    ]
except ImportError:
    pass
