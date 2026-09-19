Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ can use LMCache to
offload and reuse KV cache in vLLM workers. In :doc:`multiprocess (MP) mode
</mp/index>`, workers connect to an LMCache server running in a separate
process through ``LMCacheMPConnector``. The server keeps KV cache in CPU
memory and can use
:doc:`L2 storage adapters </mp/l2_storage/index>` for additional capacity.

The local aggregated example uses one GPU and one LMCache server:

.. code-block:: text

   Client -> Dynamo frontend -> Dynamo vLLM worker
                                       |
                               LMCacheMPConnector
                                       |
                                    CUDA IPC
                                       |
                                LMCache MP server
                                  CPU KV cache

Run locally
-----------

Complete Dynamo's `local installation
<https://docs.nvidia.com/dynamo/dev/cli/installation/install-dynamo>`_ and
start its NATS and etcd services. Use a Dynamo vLLM runtime container with
access to an NVIDIA GPU and a Dynamo checkout that includes
``launch/agg_lmcache_mp.sh`` under
``examples/backends/vllm/``.

Check the LMCache version inside the worker environment:

.. code-block:: bash

   python3 -c 'import lmcache; print(lmcache.__version__)'

The runtime must contain an LMCache build compatible with its vLLM version
and ``LMCacheMPConnector``. If you build a custom image, follow the
:doc:`installation guide </getting_started/installation>` and
:doc:`compatibility table </getting_started/compatibility>`.
When the server and worker run in separate containers, use matching LMCache
versions so they speak the same MP protocol.

From the Dynamo checkout root, start the example:

.. code-block:: bash

   cd examples/backends/vllm
   LMCACHE_L1_SIZE_GB=16 ./launch/agg_lmcache_mp.sh

The script starts ``lmcache server``, waits for its health endpoint, and
launches the Dynamo frontend and a vLLM worker serving ``Qwen/Qwen3-0.6B``.
It configures the worker with ``LMCacheMPConnector`` and ``kv_role=kv_both``.
The default ports are 8000 for inference, 5555 for the LMCache data plane,
and 8080 for LMCache health checks and metrics. Press ``Ctrl+C`` in the
launch terminal to stop the processes.

For the full environment setup, see Dynamo's `local KV cache offloading
guide <https://docs.nvidia.com/dynamo/dev/cli/kv-cache-offloading/overview>`_.
See :doc:`/mp/configuration` for LMCache server options and L2 configuration.

Check cache reuse
-----------------

In another terminal, check the server and send the same prompt twice:

.. code-block:: bash

   curl -fsS http://localhost:8080/healthcheck

   request=$(python3 - <<'PY'
   import json

   print(json.dumps({
       "model": "Qwen/Qwen3-0.6B",
       "messages": [{
           "role": "user",
           "content": "Explain how a KV cache reduces repeated computation. " * 60,
       }],
       "max_tokens": 16,
       "temperature": 0,
   }))
   PY
   )

   for run in 1 2; do
       curl -fsS http://localhost:8000/v1/chat/completions \
           -H 'Content-Type: application/json' \
           -d "$request"
   done

Inspect the LMCache lookup metrics:

.. code-block:: bash

   curl -fsS http://localhost:8080/metrics | grep '^lmcache_mp_lookup'

An increase in ``lmcache_mp_lookup_hit_tokens_total`` shows that LMCache
found cached tokens. vLLM's GPU prefix cache may satisfy repeated requests
before LMCache is queried, so the response's ``cached_tokens`` value alone
does not identify which cache supplied the hit. Use the
:doc:`LMCache metrics </mp/observability/metrics>` to check LMCache activity
and a representative workload to measure TTFT.

Deploy on Kubernetes
--------------------

Follow Dynamo's `LMCache MP deployment guide
<https://docs.nvidia.com/dynamo/dev/kubernetes/kv-cache-offloading/deploy-lm-cache-mp>`_
for the platform installation, image versions, and worker manifest. The
:doc:`LMCache operator </mp/operator>` creates a server DaemonSet from an
``LMCacheEngine`` resource, so workers on each node can share that node's
cache server.

Create the ``LMCacheEngine`` before the Dynamo workers. The operator creates
a ``<engine-name>-connection`` ConfigMap containing
``kv-transfer-config.json``. Workers mount this file and pass it to
``--kv-transfer-config``. Keep the engine, workers, and any referenced
Secrets in the same namespace.

The `examples in this repository
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration>`_
include aggregated and disaggregated manifests. These use Dynamo's
``nvidia.com/v1alpha1`` API; choose manifests that match your installed
Dynamo operator. Replace each ``my-tag`` placeholder with a compatible
image tag. The LMCache server image and Dynamo worker image must contain
matching LMCache versions.

The bundled Dynamo manifests use ``hostIPC: true`` and disable Dynamo's
separate shared-memory mount for CUDA IPC. For deployments using isolated
IPC, follow :doc:`/mp/deployment` and configure both the server and worker
consistently.

Prefill/decode disaggregation
-----------------------------

The repository's ``disagg_lmcache_mp.sh`` example runs a prefill worker and
a decode worker on two GPUs in one node. Both connect to the same LMCache
MP server. To use this recipe, copy the script from
``examples/dynamo_integration/launch/`` into the Dynamo checkout's
``examples/backends/vllm/launch/`` directory, where its shared shell helpers
are available. Then run it from ``examples/backends/vllm/``:

.. code-block:: bash

   ./launch/disagg_lmcache_mp.sh

This recipe depends on the shared server. A per-node DaemonSet alone does
not make the cache available across nodes. For a deployment that transfers
KV cache between prefill and decode nodes, follow Dynamo's
`disaggregated serving guide
<https://docs.nvidia.com/dynamo/dev/cli/disaggregated-serving/overview>`_ and select
the connectors required by that recipe.

KV-aware routing
----------------

The MP connector can pass completed cache stores to vLLM as ``BlockStored``
events with ``medium="CPU"``. Dynamo can consume these events when KV-aware
routing is configured. This requires an LMCache build containing
`MP connector KV event support
<https://github.com/LMCache/LMCache/pull/5076>`_ on the worker; older images
may support offloading without publishing these events.

Enable vLLM KV events and Dynamo KV-aware routing explicitly. Match the
LMCache chunk size to the routing block size, set
``lmcache.mp.hash_algorithm`` to the server's hash algorithm. When using
``builtin`` hashing, set the same ``PYTHONHASHSEED`` across workers, servers,
and the coordinator if one is deployed. The local MP launch scripts above
do not enable KV-aware routing. See Dynamo's `routing guide
<https://docs.nvidia.com/dynamo/dev/cli/kv-aware-routing/overview>`_ for router setup.

.. warning::

   The MP connector publishes store events only. It does not publish
   ``BlockRemoved`` or ``AllBlocksCleared`` when the MP host cache evicts or
   clears entries. Use these host-cache routing hints only when entries
   remain cached during the routing window; otherwise the router can retain
   stale cache locations.
