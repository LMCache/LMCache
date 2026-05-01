.. _tensorrt_llm_integration:

TensorRT-LLM
============

LMCache integrates with NVIDIA TensorRT-LLM via TRT-LLM's
**KV Cache Connector** API. The connector ABC ships in
``tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector`` and
lets external KV stores hook the engine lifecycle: lookup before
scheduling, retrieve before the forward pass, store after.

Two modes are available:

- **In-process** (``connector: lmcache``) — LMCache runs as a
  singleton inside the TRT-LLM process. Simplest setup; no extra
  service to manage.
- **Multi-process** (``connector: lmcache-mp``) — LMCache runs as a
  standalone ZMQ server. Multiple TRT-LLM workers on the same node
  can share the cache, and the cache survives a TRT-LLM crash.

Requirements
------------

- TensorRT-LLM 1.2.0 or newer (the ``KvCacheConnector`` ABC was added
  in 1.2.0).
- LMCache built with the ``c_ops`` extension. Verify with
  ``python -c "import lmcache.c_ops"``.
- For multi-process mode: ``cuda-python`` and ``cupy`` (used by the
  raw CUDA-IPC wrapper that publishes the TRT-LLM pool tensor across
  process boundaries).

The TRT-LLM connector preset registry resolves both shorthand names
to LMCache's adapter modules (see
`NVIDIA/TensorRT-LLM PR #12626 <https://github.com/NVIDIA/TensorRT-LLM/pull/12626>`_).

In-process mode
---------------

1. Set ``LMCACHE_CONFIG_FILE`` (or individual ``LMCACHE_*`` env vars)
   so ``LMCacheEngineConfig`` can be constructed at startup. A
   minimal CPU-offload config:

   .. code-block:: bash

      export PYTHONHASHSEED=0  # required — chunk hashing depends on stable hash()
      export LMCACHE_CHUNK_SIZE=256
      export LMCACHE_LOCAL_CPU=True
      export LMCACHE_MAX_LOCAL_CPU_SIZE=2.0  # GiB

2. Pass ``connector: lmcache`` to ``KvCacheConnectorConfig`` when
   building the LLM:

   .. code-block:: python

      from tensorrt_llm import LLM, SamplingParams
      from tensorrt_llm.llmapi.llm_args import (
          KvCacheConfig, KvCacheConnectorConfig,
      )

      llm = LLM(
          model="Qwen/Qwen2-1.5B-Instruct",
          backend="pytorch",
          kv_cache_config=KvCacheConfig(enable_block_reuse=True),
          kv_connector_config=KvCacheConnectorConfig(connector="lmcache"),
      )

      sp = SamplingParams(max_tokens=64)
      out = llm.generate(["Your prompt here"], sp)
      print(out[0].outputs[0].text)

That's the whole integration on the user side. The lifecycle hooks
(``register_kv_caches``, ``start_load_kv``, ``wait_for_save``,
``get_num_new_matched_tokens``) are wired automatically by TRT-LLM
based on the preset.

Multi-process mode
------------------

1. Start the LMCache server in its own process:

   .. code-block:: bash

      python -m lmcache.v1.multiprocess.server \
          --host 0.0.0.0 --port 5555 \
          --l1-size-gb 10 --eviction-policy LRU \
          --max-workers 4 --chunk-size 256

2. Point TRT-LLM at the server via ``server_url``:

   .. code-block:: python

      llm = LLM(
          model="Qwen/Qwen2-1.5B-Instruct",
          backend="pytorch",
          kv_cache_config=KvCacheConfig(enable_block_reuse=True),
          kv_connector_config=KvCacheConnectorConfig(
              connector="lmcache-mp",
              server_url="tcp://localhost:5555",
          ),
      )

   The server URL can also come from the ``LMCACHE_SERVER_URL`` env
   var. The Unix-socket form ``ipc:///tmp/lmcache.sock`` is a sensible
   default for single-host deployments.

The MP adapter shares the TRT-LLM KV pool with the server using a
raw CUDA-IPC wrapper (``RawCudaIPCWrapper``). PyTorch's standard
storage-IPC path raises on TRT-LLM's pool because the buffer is
allocated outside PyTorch's caching allocator (``at::for_blob`` over
``cudaMalloc``); the wrapper bypasses that path with
``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``.

Explicit connector configuration
--------------------------------

If you need to bypass the preset registry — e.g. you are pinning a
custom subclass of the adapter — point ``connector_module`` and the
class names directly:

.. code-block:: yaml

   # In-process
   kv_connector_config:
     connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
     connector_scheduler_class: LMCacheKvConnectorScheduler
     connector_worker_class: LMCacheKvConnectorWorker

   # Multi-process
   kv_connector_config:
     connector_module: lmcache.integration.tensorrt_llm.tensorrt_mp_adapter
     connector_scheduler_class: LMCacheMPKvConnectorScheduler
     connector_worker_class: LMCacheMPKvConnectorWorker
     server_url: tcp://localhost:5555

Verifying LMCache is the source of cache hits
---------------------------------------------

TRT-LLM has its own GPU block reuse, so a matching second-request
output does not by itself prove LMCache contributed. To force the
issue:

- Size TRT-LLM's pool small enough that the first prompt's blocks
  must evict before the third request runs:
  ``KvCacheConfig(max_tokens=512, enable_block_reuse=True)``.
- Send three requests: a long prompt (>512 tokens), a different
  long prompt that fills the now-tiny pool, then the original prompt
  again.

You should see DEBUG-level lines like:

.. code-block:: text

   LMCache TRT-LLM scheduler: req N ... lmcache_cached=256 new_matched=192
   Retrieved 256 out of 382 required tokens

on the third request. ``lmcache_cached`` reports how many tokens
LMCache had cached; ``new_matched`` is how many additional tokens
LMCache supplied beyond what TRT-LLM had already matched in its GPU
pool. Both should be non-zero on a real LMCache hit.

Configuration reference
-----------------------

The TRT-LLM adapter does not introduce new LMCache config keys. It
reads :class:`LMCacheEngineConfig` the same way the vLLM adapter
does: ``LMCACHE_CONFIG_FILE`` for a YAML file, otherwise individual
``LMCACHE_*`` environment variables. See
:doc:`../getting_started/installation` for the full configuration
surface.
