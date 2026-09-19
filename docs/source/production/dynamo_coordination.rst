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

Prerequisites
~~~~~~~~~~~~~

- Install the :doc:`LMCache Operator </mp/operator>`. It deploys and
  manages LMCache servers and creates the connection configuration used
  by the workers.
- Install `Dynamo
  <https://docs.nvidia.com/dynamo/dev/kubernetes/installation/install-dynamo#install-the-dynamo-platform>`_.
  Its Kubernetes operator manages the frontend and inference workers
  defined in a ``DynamoGraphDeployment`` resource.

Deploy the LMCache server
~~~~~~~~~~~~~~~~~~~~~~~~~

``lmcache_engine.yaml`` defines an ``LMCacheEngine`` custom resource with
16 GiB of CPU cache per server. The operator creates a server DaemonSet,
a Service, and a ``lmcache-mp-connection`` ConfigMap for the workers.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/lmcache_engine.yaml
   :language: yaml
   :caption: lmcache_engine.yaml

Apply the resource:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/lmcache_engine.yaml

Aggregated serving
~~~~~~~~~~~~~~~~~~

``agg_lmcache_mp.yaml`` defines a ``DynamoGraphDeployment`` with a frontend
and one vLLM worker. The worker serves ``Qwen/Qwen3-0.6B`` on one GPU and
handles both prefill and decode. Both containers use
``nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2``, which includes the same
LMCache 0.5.2 version as the cache server.

The worker mounts ``lmcache-mp-connection`` at ``/etc/lmcache`` and reads
``kv-transfer-config.json`` through ``--kv-transfer-config``. This sets up
``LMCacheMPConnector`` to connect to the server on its node.
``hostIPC: true`` and ``sharedMemory.disabled: true`` let the worker use
the same shared memory as the server.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml
   :language: yaml
   :caption: agg_lmcache_mp.yaml

Apply the aggregated deployment:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml

Disaggregated serving
~~~~~~~~~~~~~~~~~~~~~

``disagg_lmcache_mp.yaml`` defines a ``DynamoGraphDeployment`` with a
frontend and separate prefill and decode workers. Each worker uses one
GPU, with ``--disaggregation-mode`` set to ``prefill`` or ``decode``.
Both workers use the same image and connection ConfigMap as the
aggregated example and connect to the LMCache server created above.

Use a cluster with a single GPU node and at least two GPUs so both
workers connect to the same server. The manifest does not force worker
co-location or configure cache sharing between nodes.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml
   :language: yaml
   :caption: disagg_lmcache_mp.yaml

To use disaggregated serving, apply this manifest instead of the
aggregated deployment:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml
