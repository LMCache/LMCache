Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ is an open-source,
datacenter-scale inference stack. It orchestrates inference engines such
as vLLM, SGLang, and TensorRT-LLM across multiple nodes. LMCache provides a
KV cache layer that stores cache beyond GPU memory for reuse across
requests.

Local
-----

Start NATS and etcd
~~~~~~~~~~~~~~~~~~~

If you are deploying Dynamo locally, start NATS and etcd first.
On the host, run the included Compose file from the root of the LMCache
repository:

.. code-block:: bash

   docker compose -f examples/dynamo_integration/local/docker-compose.yml up -d

Start the Dynamo container
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Start LMCache and Dynamo
~~~~~~~~~~~~~~~~~~~~~~~~

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

Run with a script
~~~~~~~~~~~~~~~~~

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
16 GiB of CPU cache per server. The operator creates:

- A DaemonSet that runs one LMCache server on each eligible node.
- The ``lmcache-mp`` Service, which routes workers to the server on their
  own node.
- The ``lmcache-mp-connection`` ConfigMap. Its ``kv-transfer-config.json``
  entry contains the ``LMCacheMPConnector`` settings, including the
  Service address and server port.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/lmcache_engine.yaml
   :language: yaml
   :caption: lmcache_engine.yaml

Apply the resource:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/lmcache_engine.yaml

Aggregated serving
~~~~~~~~~~~~~~~~~~

`agg_lmcache_mp.yaml
<https://github.com/LMCache/LMCache/blob/dev/examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml>`_
defines a ``DynamoGraphDeployment`` with a frontend
and one vLLM worker. The worker serves ``Qwen/Qwen3-0.6B`` on one GPU and
handles both prefill and decode.

The worker mounts ``lmcache-mp-connection`` at ``/etc/lmcache``, making
the configuration available as ``/etc/lmcache/kv-transfer-config.json``.
At startup, it passes the file's JSON contents to ``--kv-transfer-config``
to connect to the LMCache server on its node.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml
   :language: yaml
   :caption: agg_lmcache_mp.yaml

Apply the aggregated deployment:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/agg_lmcache_mp.yaml

Disaggregated serving
~~~~~~~~~~~~~~~~~~~~~

`disagg_lmcache_mp.yaml
<https://github.com/LMCache/LMCache/blob/dev/examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml>`_
starts a frontend and separate prefill and
decode workers. Use a cluster with one GPU node and at least two GPUs.
Each worker uses one GPU and connects to the same LMCache server.

.. literalinclude:: ../../../examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml
   :language: yaml
   :caption: disagg_lmcache_mp.yaml

Apply the disaggregated deployment:

.. code-block:: bash

   kubectl apply -n default -f examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml
