.. _lookup:

Lookup the KV cache
===================

The ``lookup`` interface is defined as the following: 

.. code-block:: python

    lookup(tokens: List[int]) -> layout_info: Dict[str, Tuple[str, int]]

The function takes a list of tokens as input and returns a dictionary containing the layout information for each token. 
The layout information is represented as a mapping between ``instance_id`` and a tuple of ``(location, matched_prefix_length)``.

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

Finally, we can send a ``lookup`` request to the lmcache controller:

.. code-block:: bash

    curl -X POST http://localhost:9000/lookup \
    -H "Content-Type: application/json" \
    -d '{
        "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
    }'

We should be able to see the response like this:

.. code-block:: text

    {"lmcache_default_instance_id": ["LocalCPUBackend",12]}

This means that the KV cache for the given tokens is stored in ``lmcache_default_instance``'s CPU memory and the matched prefix length is 12.