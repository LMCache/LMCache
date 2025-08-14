Caching by request
===================================

LMCache supports caching by request

For example, if you want to cache some requests and redo prefill for other requests, you can add ``caching`` field to
``kv_transfer_params`` to control whether to cache this request.

example.yaml

.. code-block:: yaml

	chunk_size: 256
	local_device: "cpu"
	local_cpu: True
	max_local_cpu_size: 10


1. Start the vllm engine at port 8000:

.. code-block:: bash

	VLLM_USE_V1=1 \
	LMCACHE_USE_EXPERIMENTAL=True \
	LMCACHE_TRACK_USAGE=false \
	LMCACHE_CONFIG_FILE=example.yaml \
	vllm serve /disc/f/models/opt-125m/ \
			   --served-model-name "facebook/opt-125m" \
			   --enforce-eager  \
			   --port 8000 \
			   --gpu-memory-utilization 0.8 \
			   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
			   --trust-remote-code

2. Send a request to vllm engine with ``kv_transfer_params: {caching: false}``:

.. code-block:: bash

	curl -X POST http://localhost:8000/v1/completions \
	  -H "Content-Type: application/json" \
	  -d '{
		"model": "facebook/opt-125m",
		"prompt": "Explain the significance of KV cache in language models." * 100,
		"max_tokens": 10,
		"kv_transfer_params": {
		  "caching": false
		}
	  }'

This request will not be cached.

3. Send a request to vllm engine with ``kv_transfer_params: {caching: true}`` or not pass the param:

.. code-block:: bash

	curl -X POST http://localhost:8000/v1/completions \
	  -H "Content-Type: application/json" \
	  -d '{
		"model": "facebook/opt-125m",
		"prompt": "Explain the significance of KV cache in language models." * 100,
		"max_tokens": 10,
		"kv_transfer_params": {
		  "caching": true
		}
	  }'

This request will be cached.

