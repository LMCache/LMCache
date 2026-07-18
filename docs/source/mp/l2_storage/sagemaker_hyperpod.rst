SageMaker HyperPod
==================

An L2 adapter backed by the **ai-toolkit** daemon of Amazon SageMaker
HyperPod, which provides shared memory access for LMCache.

**Required fields:**

- ``url``: ai-toolkit daemon URL (``sagemaker-hyperpod://host:port``).

**Optional fields:**

- ``bucket`` (string, default ``"lmcache"``): ai-toolkit cache namespace.
- ``shared_memory_name`` (string, default ``"shared_memory"``): Name of
  shared memory segment.
- ``max_concurrent_requests`` (int, default ``100``): Maximum concurrent
  HTTP requests.
- ``max_connections`` (int, default ``256``): Total HTTP connection-pool
  limit.
- ``max_connections_per_host`` (int, default ``128``): Per-host connection
  limit.
- ``timeout_ms`` (int, default ``5000``): HTTP transport timeout.
- ``lease_wait_timeout_ms`` (int, default ``1000``): How long the daemon
  may hold a lease request before answering.
- ``lease_ttl_ms`` (int, default ``30000``): Server-side lease lifetime.
- ``put_stream_chunk_bytes`` (int, default ``65536``): HTTP PUT chunk size.
- ``max_lease_size_mb`` (float, optional): Upper bound for accepted lease
  payloads.
- ``use_https`` (bool, default ``false``): Enable HTTPS instead of HTTP.

**Configuration examples:**

.. code-block:: bash

    # Basic (node IP via the Kubernetes downward API)
    --l2-adapter '{"type": "sagemaker-hyperpod", "url": "sagemaker-hyperpod://'"${NODE_IP}"':9200"}'

    # Custom namespace and faster miss detection
    --l2-adapter '{"type": "sagemaker-hyperpod", "url": "sagemaker-hyperpod://'"${NODE_IP}"':9200", "bucket": "team-a", "lease_wait_timeout_ms": 250}'

Kubernetes Deployment Requirements
-----------------------------------

The ai-toolkit daemon is node-local (hostNetwork, port 9200), so the
LMCache server must run on the same node as the daemon.

Environment Variable for Node IP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the ``NODE_IP`` environment variable to resolve the local node's IP
address:

.. code-block:: yaml

   env:
     - name: NODE_IP
       valueFrom:
         fieldRef:
           fieldPath: status.hostIP

/dev/shm Volume Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Hyperpod requires /dev/shm for high-performance shared memory
operations:

.. code-block:: yaml

   volumeMounts:
     - name: dshm
       mountPath: /dev/shm/shared_memory
       subPath: shared_memory

   volumes:
     - name: dshm
       hostPath:
         path: /dev/shm
