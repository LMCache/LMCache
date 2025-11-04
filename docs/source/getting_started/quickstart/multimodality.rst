Example: Multimodal KV Cache Support
====================================

Quick Start Example (Vision-Language Model):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are going to be running multimodal inference with ``Qwen/Qwen2-VL-2B-Instruct`` and using LMCache to speed up the TTFT after the first request.

.. note::

    This guide has been tested with **vLLM 0.11.0** and **LMCache 0.3.9** using the Qwen/Qwen2-VL-2B-Instruct model.

**Install and Serve:**

``pip install lmcache vllm openai``

.. code-block:: bash

   LMCACHE_USE_EXPERIMENTAL=True \
   vllm serve Qwen/Qwen2-VL-2B-Instruct \
       --max-model-len 1024 \
       --gpu-memory-utilization 0.5 \
       --enforce-eager \
       --limit-mm-per-prompt '{"image": 1}' \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'


**Test the setup:**

Send a simple text request to verify LMCache is working:

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "Qwen/Qwen2-VL-2B-Instruct",
           "prompt": "Describe the color blue.",
           "max_tokens": 50
       }'   


**Monitoring LMCache:**

After running the first request, check the vLLM logs to see LMCache storing the KV cache:

.. code-block:: text

   LMCache INFO: Storing KV cache for 5 out of 5 tokens (skip_leading_tokens=0) for request 0
   LMCache INFO: Stored 5 out of total 5 tokens. size: 0.0001 gb, cost 0.5948 ms, throughput: 0.2245 GB/s

Run the same request again, and you'll see LMCache retrieving the cached KV:

.. code-block:: text

   LMCache INFO: Reqid: 0, Total tokens 5, LMCache hit tokens: 0, need to load: 0
   LMCache INFO: Retrieved 5 out of 5 out of total 5 tokens

This demonstrates that LMCache successfully caches and retrieves KV values for multimodal models.

.. note::

   LMCache works with various multimodal models supported by vLLM, including vision-language and audio-language models. The KV cache is automatically managed for both text and multimodal tokens.