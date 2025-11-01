SageMaker Hyperpod Backend
===========================

Prerequisites
-------------

Create an Amazon SageMaker HyperPod cluster with tiered storage enabled by following the instructions at:

https://docs.aws.amazon.com/sagemaker/latest/dg/managed-tier-checkpointing-setup.html

This enables the ai-toolkit daemon that provides shared memory access for LMCache.

Example Configuration
---------------------

.. code-block:: yaml

   chunk_size: 256
   local_cpu: True
   max_local_cpu_size: 5
   remote_url: "sagemaker-hyperpod://$NODE_IP:9200"

Configuration Parameters
------------------------

SageMaker Hyperpod-Specific (in extra_config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **sagemaker_hyperpod_bucket**: Bucket name for KV storage namespace (default: "lmcache")
* **sagemaker_hyperpod_shared_memory_name**: Name of shared memory segment (default: "shared_memory"). Set to None to disable shared memory.
* **sagemaker_hyperpod_max_concurrent_requests**: Maximum concurrent control plane requests (default: 100, minimum: 1)
* **sagemaker_hyperpod_max_connections**: Maximum total HTTP connections in pool (default: 256, minimum: 1)
* **sagemaker_hyperpod_max_connections_per_host**: Maximum HTTP connections per host (default: 128, minimum: 1)
* **sagemaker_hyperpod_timeout_ms**: Timeout for lease acquisition requests in milliseconds (default: 5000, minimum: 100)
* **sagemaker_hyperpod_lease_ttl_s**: Server-side lease timeout in seconds (default: 30.0)
* **sagemaker_hyperpod_put_stream_chunk_bytes**: Chunk size for streaming PUT requests in bytes (default: 65536, minimum: 1024)
* **sagemaker_hyperpod_use_https**: Enable HTTPS instead of HTTP (default: False). **Note**: Ignored if ``remote_url`` already contains ``http://`` or ``https://`` protocol.
* **save_chunk_meta**: Whether to save chunk metadata with data (set False for performance)

Kubernetes Deployment Requirements
-----------------------------------

Environment Variable for Node IP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the ``NODE_IP`` environment variable to resolve the local node's IP address:

.. code-block:: yaml

   env:
     - name: NODE_IP
       valueFrom:
         fieldRef:
           fieldPath: status.hostIP

/dev/shm Volume Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Hyperpod requires /dev/shm for high-performance shared memory operations:

.. code-block:: yaml

   volumeMounts:
     - name: dshm
       mountPath: /dev/shm/shared_memory
       subPath: shared_memory

   volumes:
     - name: dshm
       hostPath:
         path: /dev/shm
