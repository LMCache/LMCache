Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ is an open-source,
datacenter-scale inference stack. It orchestrates inference engines such
as vLLM, SGLang, and TensorRT-LLM across multiple nodes. LMCache provides a
KV cache layer that stores cache beyond GPU memory for reuse across
requests.

Local
-----

If you are deploying Dynamo locally, start NATS and etcd first.
On the host, run the included Compose file from the root of the LMCache
repository:

.. code-block:: bash

   docker compose -f examples/dynamo_integration/local/docker-compose.yml up -d

This demo serves ``Qwen/Qwen3-0.6B`` on a single GPU, with 16 GiB of CPU
memory for LMCache. The ``nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2`` image
includes LMCache 0.5.2. You can find the latest image tags on `NVIDIA NGC
<https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers/vllm-runtime/-/tags>`_.

From the same directory, start the Dynamo container:

.. code-block:: bash

   docker run --rm -it --name dynamo-lmcache \
       --gpus all --network host --ipc host \
       --ulimit memlock=-1 \
       -v "$PWD:/workspace/LMCache:ro" \
       nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2 bash

This opens a shell in the container, where LMCache is already installed.
Open two more terminals on the host and enter the same container in each:

.. code-block:: bash

   docker exec -it dynamo-lmcache bash

In the three container shells, start one process in each:

.. code-block:: bash

   # Terminal 1: start the LMCache server with 16 GiB of CPU cache.
   lmcache server --l1-size-gb 16 --eviction-policy LRU \
       --port 5555 --http-port 8080

   # Terminal 2: start the inference frontend.
   python3 -m dynamo.frontend

   # Terminal 3: once LMCache is ready, start the vLLM worker and connect it to LMCache.
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
on port 5555.

You can also start the whole demo with a `script
<https://github.com/LMCache/LMCache/blob/dev/examples/dynamo_integration/local/launch_lmcache_mp.sh>`_.
On the host, run one of these commands from the root of the LMCache repository:

.. code-block:: bash

   # Aggregated: 1 GPU.
   ./examples/dynamo_integration/local/launch_lmcache_mp.sh aggregated

   # Disaggregated: 2 GPUs on the same node.
   ./examples/dynamo_integration/local/launch_lmcache_mp.sh disaggregated

Kubernetes
----------

The `Kubernetes manifests
<https://github.com/LMCache/LMCache/tree/dev/examples/dynamo_integration/kubernetes>`_
deploy the same model with Dynamo's vLLM backend. After preparing the
cluster and image tags described below, run these two commands from the
root of the LMCache repository:

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
