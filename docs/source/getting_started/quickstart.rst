.. _quickstart:

Quickstart
==========

LMCache has the same interface as vLLM (both online serving and offline inference). 
To use the online serving, you can start an OpenAI API-compatible vLLM server with LMCache via:

.. code-block:: console

    $ lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8

To use vLLM's offline inference with LMCache, just simply add lmcache_vllm before the import to vLLM components. For example

.. code-block:: python

    import lmcache_vllm.vllm as vllm
    from lmcache_vllm.vllm import LLM 

    # Load the model
    model = LLM.from_pretrained("lmsys/longchat-7b-16k")

    # Use the model
    model.generate("Hello, my name is", max_length=100)



