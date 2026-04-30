.. _modular_max_integration:

Modular MAX
===========

LMCache can serve Modular MAX KV cache requests in multiprocess (MP)
mode. MAX keeps using its existing ``--kv-connector lmcache`` flag; MP
mode is selected through ``--kv-connector-config``.

Start LMCache MP Server
-----------------------

Start a standalone LMCache server with a chunk size that is compatible
with the MAX KV page size:

.. code-block:: bash

   python -m lmcache.v1.multiprocess.server \
       --host 127.0.0.1 \
       --port 5555 \
       --chunk-size 256 \
       --l1-size-gb 10

The LMCache ``chunk-size`` must be an integer multiple of MAX
``kv_cache_page_size``. For token-mode MAX integration, LMCache fails at
startup or connector initialization if this relationship does not hold:

.. code-block:: text

   LMCache MP chunk_size must be a multiple of MAX page_size for token-mode MAX integration.

Start MAX
---------

Run MAX with prefix caching enabled and opt in to MP mode:

.. code-block:: bash

   max serve \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --enable-prefix-caching \
       --kv-connector lmcache \
       --kv-connector-config '{
         "lmcache_mode": "mp",
         "lmcache_model_name": "meta-llama/Llama-3.1-8B-Instruct",
         "lmcache_mp_host": "127.0.0.1",
         "lmcache_mp_port": 5555
       }'

``lmcache_model_name`` is required by the current MAX MP connector and
should be stable across MAX processes that are expected to share cache
entries. A model name mismatch causes cache misses because LMCache
includes the model name in its object keys.

The MAX MP connector sends text token IDs to LMCache and uses the normal
token-mode MP key path. It does not send MAX block hashes as token IDs.
Multimodal/image prompts are not supported in the first token-mode
implementation and fail clearly until a cross-engine image-token contract
is defined.

Configuration
-------------

``lmcache_mode``
   Omit this key, or set it to ``"local"``, to keep the existing
   in-process MAX LMCache connector. Set it to ``"mp"`` to use the
   LMCache MP connector.

``lmcache_model_name``
   Required stable LMCache key namespace. Do not use random per-process
   names unless you want isolated caches.

``lmcache_mp_host`` and ``lmcache_mp_port``
   MP server host and port. Defaults are ``127.0.0.1`` and ``5555``.

``lmcache_mp_server_url``
   Full ZMQ URL. When present, it overrides host and port.

``lmcache_mp_timeout_s``
   Request timeout in seconds.

``lmcache_allow_host_staging``
   Host staging fallback opt-in. The default is ``false``.

Backends and Layout
-------------------

The MAX registration boundary uses a backend-neutral device buffer
descriptor. CUDA IPC is the first implemented backend. ROCm/AMD is not
silently routed through CUDA; unsupported backends fail with a targeted
error:

.. code-block:: text

   LMCache MP for Modular MAX does not yet support backend 'rocm'. Implement HipIpcBufferImporter or enable explicit host staging fallback.

LMCache supports the Modular MAX physical KV layout:

.. code-block:: text

   [NB, KVDIM, NL, BS, NH, HS]

where ``KVDIM`` is ``2`` for normal K/V and ``1`` for MLA,
``BS`` is the MAX page size, and ``NB`` is the total number of device
blocks.

Troubleshooting
---------------

Chunk/page mismatch
   Set LMCache ``--chunk-size`` to a multiple of MAX
   ``kv_cache_page_size``.

Unsupported layout
   Verify the registered MAX buffer shape is
   ``[NB, KVDIM, NL, BS, NH, HS]`` and that ``KVDIM`` is ``1`` or ``2``.

Unsupported backend
   CUDA IPC is supported first. ROCm/HIP requires a HIP IPC importer, or
   explicit host staging once that fallback is implemented.

MP server unavailable
   Check the server URL, host, port, and firewall rules. The connector
   queries ``GET_CHUNK_SIZE`` during initialization and fails if the
   server does not respond.

Unexpected cache misses
   Confirm every MAX process uses the same ``lmcache_model_name`` and
   the same LMCache chunk size/page-size relationship.

Multimodal request rejected
   The first MAX MP token-mode connector supports text prompts only.
   Use in-process mode for multimodal prompts until LMCache and MAX share
   a stable image-token convention.

MP e2e test environment
   Do not inject a separate uv LMCache or PyTorch installation into a running
   Bazel/MAX Python process. For full MP e2e validation, start LMCache and
   MAX as separate processes in one compatible Torch/CUDA runtime environment.
