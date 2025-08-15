.. _clear:

Clear the KV cache
==================

The ``clear`` interface is defined as the following: 

.. code-block:: python

    clear(instance_id: str, tokens: Optional[List[int]], locations: Optional[List[str]]) -> success: bool

The function takes the ``instance_id`` and optionally ``tokens`` and ``locations`` as input. 
The return value is a boolean indicating whether the ``clear`` operation was successful or not.
If ``tokens`` and ``locations`` are not provided, all the KV caches on the given instance will be cleared.

Example usage:
---------------------------------------

First, we need to start the lmcache controller at port 9000 and the monitor at port 9001:

.. code-block:: bash

    python -m lmcache.v1.api_server --port 9000 --monitor-port 9001

Second, we need a yaml file ``example.yaml`` to properly configure the lmcache instance:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5

    # cache controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_default_instance"
    controller_url: "localhost:9001"
    distributed_url: "localhost:8002"
    lmcache_worker_port: 8001

Third, we need to start the vllm/lmcache instance:

.. code-block:: bash

    LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Then, we can send a request to vllm: 

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 10
    }'

We send a ``clear`` request to the lmcache controller:

.. code-block:: bash

    curl -X POST http://localhost:9000/clear \
    -H "Content-Type: application/json" \
    -d '{
        "instance_id": "lmcache_default_instance_id",
        "location": "LocalCPUBackend"
    }'

We should be able to see the response like this:

.. code-block:: text

    {"success": True}

This indicates all the KV caches on the ``lmcache_default_instance`` have been cleared.

We can further verify this by sending a ``lookup`` request to the lmcache controller:

.. code-block:: bash

    curl -X POST http://localhost:9000/lookup \
    -H "Content-Type: application/json" \
    -d '{
        "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
    }'

We should be able to see an empty the response, indicating the KV cache for the given tokens has been cleared.