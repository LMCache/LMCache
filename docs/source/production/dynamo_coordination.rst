Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ is an open-source,
datacenter-scale inference stack. It orchestrates inference engines such
as vLLM, SGLang, and TensorRT-LLM across multiple nodes. LMCache provides a
KV cache layer that stores cache beyond GPU memory for reuse across
requests.

Local
-----

We recommend starting with a single NVIDIA GPU and Dynamo's
``vllm-runtime`` container. The commands below use vLLM to serve
``Qwen/Qwen3-0.6B``.

Start NATS and etcd first if they are not already running. Run this command
from the Dynamo checkout root on the host:

.. code-block:: bash

   docker compose -f dev/docker-compose.yml up -d

Use a runtime image containing an LMCache build compatible with its vLLM
version; see the :doc:`compatibility table </getting_started/compatibility>`.
Give the container access to the GPU and make sure it can reach NATS and
etcd, for example through Docker's ``--network host`` option. Leave
``PROMETHEUS_MULTIPROC_DIR`` unset so Dynamo can manage it. This setup is
needed whether you start the processes manually or use the launch scripts
below.

To start the processes manually, open three terminal sessions in the same
container and run one command in each:

.. code-block:: bash

   # Terminal 1: start the LMCache server.
   lmcache server --l1-size-gb 16 --eviction-policy LRU \
       --port 5555 --http-port 8080

   # Terminal 2: start the inference frontend.
   python3 -m dynamo.frontend

   # Terminal 3: start the vLLM worker after LMCache is ready.
   DYN_SYSTEM_PORT=8081 python3 -m dynamo.vllm \
       --model Qwen/Qwen3-0.6B \
       --enforce-eager \
       --max-model-len 4096 \
       --max-num-seqs 2 \
       --disable-hybrid-kv-cache-manager \
       --kv-transfer-config '{
         "kv_connector": "LMCacheMPConnector",
         "kv_role": "kv_both",
         "kv_connector_extra_config": {"lmcache.mp.port": 5555}
       }'

The frontend accepts inference requests on port 8000. The worker runs the
model and sends KV cache operations to LMCache on port 5555. LMCache keeps
up to 16 GiB of KV cache in CPU memory and evicts entries using LRU. Its
health and metrics endpoints use port 8080; ``DYN_SYSTEM_PORT=8081`` keeps
the worker's HTTP port separate.

``kv_role=kv_both`` allows the worker to store and retrieve KV cache.
``lmcache.mp.port`` must match the server's ``--port``. The example limits
requests to 4,096 tokens and two concurrent sequences, and disables vLLM's
hybrid KV cache manager.

Before starting the worker, confirm that LMCache is ready:

.. code-block:: bash

   curl -fsS http://localhost:8080/healthcheck

Use the launch scripts
~~~~~~~~~~~~~~~~~~~~~~

Once the container is ready and NATS and etcd are running, you can use the
`launch scripts
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration/launch>`_
to start LMCache, the Dynamo frontend, and the vLLM workers together. The
scripts wait for LMCache to become healthy and stop the processes when you
press ``Ctrl+C``. They also set a GPU KV cache memory budget for the example
model. Stop any manually launched processes before running a script to
free their ports and GPU memory.

Copy the scripts into the Dynamo checkout inside the runtime container.
Replace the paths below with the locations of your checkouts:

.. code-block:: bash

   cp /path/to/LMCache/examples/dynamo_integration/launch/*_lmcache_mp.sh \
       /path/to/dynamo/examples/backends/vllm/launch/
   cd /path/to/dynamo/examples/backends/vllm

The scripts load helpers from ``examples/common/`` relative to their own
location, so they must be placed in this directory. Choose one mode:

.. tab-set::

   .. tab-item:: Aggregated (1 GPU)

      .. code-block:: bash

         LMCACHE_L1_SIZE_GB=16 ./launch/agg_lmcache_mp.sh

      One worker handles both prefill and decode.

   .. tab-item:: Disaggregated (2 GPUs)

      .. code-block:: bash

         LMCACHE_L1_SIZE_GB=16 ./launch/disagg_lmcache_mp.sh

      The decode worker uses GPU 0 and the prefill worker uses GPU 1.
      Both connect to one LMCache server on the same node. This script
      does not configure KV transfer between nodes.

``LMCACHE_L1_SIZE_GB`` sets the CPU cache capacity. The scripts also accept
``MAX_MODEL_LEN`` and ``MAX_CONCURRENT_SEQS``; their defaults are 4096 and 2.
These examples use Dynamo's default routing without enabling KV-aware
routing.

Check the deployment
~~~~~~~~~~~~~~~~~~~~

Send a request to the frontend:

.. code-block:: bash

   curl -fsS http://localhost:8000/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{
         "model": "Qwen/Qwen3-0.6B",
         "messages": [{"role": "user", "content": "What is a KV cache?"}],
         "max_tokens": 32
       }'

This checks that inference works. To check LMCache reuse, send a long
prompt more than once and inspect the server's lookup metrics:

.. code-block:: bash

   curl -fsS http://localhost:8080/metrics | grep '^lmcache_mp_lookup'

An increase in ``lmcache_mp_lookup_hit_tokens_total`` shows an LMCache
hit. vLLM can also serve repeated prompts from its GPU prefix cache, so
``cached_tokens`` in an inference response alone does not identify an
LMCache hit.

Kubernetes
----------

The `Kubernetes manifests
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration/deploy>`_
deploy the same model with Dynamo's vLLM backend. After preparing the
cluster and image tags described below, run these two commands from the
LMCache checkout root:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/deploy/lmcache_engine.yaml
   kubectl apply -n default -f examples/dynamo_integration/deploy/agg_lmcache_mp.yaml

The first command creates the shared cache service. The second creates a
Dynamo frontend and one vLLM worker. The LMCache operator generates the
connection ConfigMap that the worker mounts at startup.

Prepare the cluster and manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cluster needs GPU support, the :doc:`LMCache operator </mp/operator>`,
and the Dynamo platform installed with its infrastructure services. The
Dynamo operator must serve ``nvidia.com/v1alpha1``, the API used by these
examples. The aggregated deployment needs one GPU.

Replace ``my-tag`` in each manifest before applying it:

- In ``lmcache_engine.yaml``, choose a tag for ``lmcache/vllm-openai``.
- In the Dynamo manifest, set the
  ``nvcr.io/nvidia/ai-dynamo/vllm-runtime`` tag for the frontend and every
  worker. The worker's bundled LMCache must be compatible with the server's
  MP protocol. Use matching LMCache versions on both sides.

Both Dynamo components reference ``hf-token-secret``. Set ``HF_TOKEN`` in
your shell and create the Secret before applying the Dynamo manifest:

.. code-block:: bash

   kubectl create secret generic hf-token-secret -n default \
       --from-literal=HF_TOKEN="$HF_TOKEN"

Keep the Secret, ``LMCacheEngine``, and Dynamo deployment in the same
namespace. ``lmcache_engine.yaml`` explicitly sets ``namespace: default``;
edit that field as well as the commands if you use another namespace.

How the YAML connects the services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lmcache_engine.yaml`` defines the cache server image and CPU cache size:

.. literalinclude:: ../../../examples/dynamo_integration/deploy/lmcache_engine.yaml
   :language: yaml

The operator creates a server DaemonSet, a Service, and the
``lmcache-mp-connection`` ConfigMap. Workers use the server on their own
node, which lets multiple workers on that node share the CPU cache.

The `aggregated Dynamo manifest
<https://github.com/LMCache/LMCache/blob/dev/examples/dynamo_integration/deploy/agg_lmcache_mp.yaml>`_
contains a ``Frontend`` service and a ``VllmDecodeWorker`` service. Despite
its name, this worker handles both prefill and decode because no
``--disaggregation-mode`` is set.

.. list-table:: Worker configuration
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Purpose
   * - ``resources.limits.gpu: "1"``
     - Allocates one GPU to the worker.
   * - ``lmcache-mp-connection`` volume
     - Mounts the operator's ConfigMap at ``/etc/lmcache``.
   * - ``--kv-transfer-config``
     - Reads ``/etc/lmcache/kv-transfer-config.json`` to configure the
       connector and server endpoint.
   * - ``PYTHONHASHSEED: "0"``
     - Makes builtin token hashing deterministic across processes.
   * - ``hostIPC: true`` and ``sharedMemory.disabled: true``
     - Uses the host IPC namespace and avoids a separate Dynamo
       ``/dev/shm`` mount in this example.

The worker reads its IPC settings from the generated connection JSON.
The server and worker must use the same IPC mode; see
:doc:`/mp/deployment` for the shared-memory and isolated IPC settings.

For separate prefill and decode workers, use
``disagg_lmcache_mp.yaml`` in the second ``kubectl apply`` command instead
of ``agg_lmcache_mp.yaml``. This manifest adds ``VllmPrefillWorker`` and
sets ``--disaggregation-mode prefill`` and ``--disaggregation-mode decode``
on the two workers. Each requests one GPU and mounts the same connection
ConfigMap.

Run this disaggregated example on a single GPU node with at least two
GPUs so both workers use the same MP server. The manifest does not force
worker co-location or configure cache sharing between nodes.

Verify the deployment
~~~~~~~~~~~~~~~~~~~~~

Check the cache engine, connection ConfigMap, and Dynamo resources:

.. code-block:: bash

   kubectl -n default get lmcacheengine lmcache-mp
   kubectl -n default get configmap lmcache-mp-connection
   kubectl -n default get dynamographdeployment vllm-agg-lmcache
   kubectl -n default get pods

For the disaggregated example, the Dynamo resource is named
``vllm-disagg-lmcache``. Once the pods are ready, replace
``FRONTEND_POD_NAME`` with the frontend pod name and forward its HTTP port:

.. code-block:: bash

   kubectl -n default port-forward pod/FRONTEND_POD_NAME 8000:8000

Send the same request shown in the Local section to check inference.
