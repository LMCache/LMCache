Quickstart
==========

LMCache v1
----------

For LMCache v1, you can start the LMCache server with the following command:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=./lmcache_config.yaml \
    LMCACHE_USE_EXPERIMENTAL=True vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'


.. note::
    For LMCache v1, please refer to the examples in the :ref:`v1_index` section. 
    LMCache v1 can be directly run with the ``vllm serve`` command.

LMCache v0
-----------

For LMCache v0, you can start the LMCache server with the following command:

LMCache has the same interface as vLLM (both online serving and offline inference). 
To use the online serving, you can start an OpenAI API-compatible vLLM server with LMCache via:

.. code-block:: console

    $ lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8

To use vLLM's offline inference with LMCache, just simply add ``lmcache_vllm`` before the import to the vLLM components. For example

.. code-block:: python

    import lmcache_vllm.vllm as vllm
    from lmcache_vllm.vllm import LLM 

    # Load the model
    model = LLM.from_pretrained("lmsys/longchat-7b-16k")

    # Use the model
    model.generate("Hello, my name is", max_length=100)






