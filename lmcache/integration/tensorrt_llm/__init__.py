# SPDX-License-Identifier: Apache-2.0
"""LMCache integration for NVIDIA TensorRT-LLM via the KV Cache Connector API.

Public connector classes for the TRT-LLM ``kv_connector_config`` YAML key::

    kv_connector_config:
      connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
      connector_scheduler_class: LMCacheKvConnectorScheduler
      connector_worker_class: LMCacheKvConnectorWorker

See ``tensorrt_adapter.py`` for implementation details and ``README.md`` for
installation and configuration instructions.
"""

# tensorrt_llm is an optional dependency.  Guard the import so that
# ``import lmcache.integration.tensorrt_llm`` does not crash when the
# package is absent (e.g. during core-LMCache unit tests or doc builds).
try:
    # Local
    from lmcache.integration.tensorrt_llm.tensorrt_adapter import (
        LMCacheKvConnectorScheduler,
        LMCacheKvConnectorWorker,
        destroy_engine,
    )

    __all__ = [
        "LMCacheKvConnectorScheduler",
        "LMCacheKvConnectorWorker",
        "destroy_engine",
    ]
except ImportError:
    __all__ = []
