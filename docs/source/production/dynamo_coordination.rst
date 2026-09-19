Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ is an open-source,
datacenter-scale inference stack. It orchestrates inference engines such
as vLLM, SGLang, and TensorRT-LLM across multiple nodes. LMCache provides a
KV cache layer that stores cache beyond GPU memory for reuse across
requests.

Local
-----

If you are deploying Dynamo locally, start NATS and etcd first. Run the
included Compose file from the LMCache checkout root on the host:

.. code-block:: bash

   docker compose -f examples/dynamo_integration/local/docker-compose.yml up -d

We use ``Qwen/Qwen3-0.6B`` on a single GPU for this demo. Run the commands
below inside a Dynamo ``vllm-runtime`` container with LMCache installed.
Give the container GPU access and use ``--network host`` so it can reach
NATS and etcd.

Open three terminals in the same container and start one process in each:

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

The frontend accepts requests on port 8000. The vLLM worker uses
``LMCacheMPConnector`` to store and retrieve KV cache through the server
on port 5555. This example gives LMCache 16 GiB of CPU memory.

Before starting the worker, confirm that LMCache is ready:

.. code-block:: bash

   curl -fsS http://localhost:8080/healthcheck

Use the launch scripts
~~~~~~~~~~~~~~~~~~~~~~

To start the same services from one terminal, use the `launch scripts
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration/local>`_
inside the runtime container. NATS and etcd must already be running. Stop
any manually launched LMCache and Dynamo processes before switching to a
script.

The scripts use Dynamo's launch helpers, so copy them into the Dynamo
checkout as shown below. Replace the paths with your checkout locations:

.. code-block:: bash

   cp /path/to/LMCache/examples/dynamo_integration/local/*_lmcache_mp.sh \
       /path/to/dynamo/examples/backends/vllm/launch/
   cd /path/to/dynamo/examples/backends/vllm

Choose one mode:

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

The scripts wait for LMCache to become healthy before starting the
workers. Press ``Ctrl+C`` to stop the serving processes. NATS and etcd
continue running through Docker Compose.

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
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration/kubernetes>`_
deploy the same model with Dynamo's vLLM backend. After preparing the
cluster and image tags described below, run these two commands from the
LMCache checkout root:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/lmcache_engine.yaml
   kubectl apply -n default -f examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml

The first command creates the shared cache service. The second creates a
Dynamo frontend and one vLLM worker. The LMCache operator generates the
connection ConfigMap that the worker mounts at startup.

Prepare the cluster and manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a GPU cluster with the Dynamo platform and
:doc:`LMCache operator </mp/operator>` installed. These manifests require
a Dynamo operator that serves ``nvidia.com/v1alpha1``. The aggregated demo
needs one GPU.

Replace ``my-tag`` in each manifest before applying it:

- In ``lmcache_engine.yaml``, choose a tag for ``lmcache/vllm-openai``.
- In the Dynamo manifest, set the
  ``nvcr.io/nvidia/ai-dynamo/vllm-runtime`` tag for the frontend and every
  worker.

Use server and worker images with matching LMCache versions.

Both Dynamo components reference ``hf-token-secret``. Set ``HF_TOKEN`` in
your shell and create the Secret before applying the Dynamo manifest:

.. code-block:: bash

   kubectl create secret generic hf-token-secret -n default \
       --from-literal=HF_TOKEN="$HF_TOKEN"

The examples use the ``default`` namespace. If you change it, update both
the manifests and commands so the Secret, cache engine, and Dynamo
deployment stay in the same namespace.

How the YAML connects the services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lmcache_engine.yaml`` defines the cache server image and CPU cache size:

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/lmcache_engine.yaml
   :language: yaml

The operator creates a server DaemonSet, a Service, and the
``lmcache-mp-connection`` ConfigMap. Workers use the server on their own
node, which lets multiple workers on that node share the CPU cache.

The `aggregated Dynamo manifest
<https://github.com/LMCache/LMCache/blob/dev/examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml>`_
starts a frontend and one ``VllmDecodeWorker`` that handles both prefill
and decode. The worker mounts the connection ConfigMap at ``/etc/lmcache``
and reads ``kv-transfer-config.json`` through ``--kv-transfer-config``.
This configures ``LMCacheMPConnector`` to connect to the cache server.

For separate prefill and decode workers, use
``disagg_lmcache_mp.yaml`` in the second ``kubectl apply`` command instead
of ``agg_lmcache_mp.yaml``. It launches one worker for prefill and one for
decode, each using one GPU and the same connection ConfigMap.

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
