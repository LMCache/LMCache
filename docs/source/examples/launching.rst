.. _launching:

Launching
================

LMCache can be launched in multiple ways. Here are some examples:

How to:
----------------

* Launch a single vLLM instance with LMCache?

.. code-block:: python

   import time
   from lmcache_vllm.vllm import LLM, SamplingParams

   # Sample prompts.
   prompts = [
      "Hello, my name is",
      "The most popular sports in the world is",
      "The father of computer is",
      "The future of AI is",
   ]
   # Create a sampling params object.
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

   # Create an LLM.
   llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
            gpu_memory_utilization=0.8)
   # Generate texts from the prompts. The output is a list of RequestOutput objects
   # that contain the prompt, generated text, and other information.
   t1 = time.perf_counter()
   outputs = llm.generate(prompts, sampling_params)
   t2 = time.perf_counter()
   # Print the outputs.
   for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   print(f"Total Time: {t2 - t1} seconds")

* Launch multiple vLLM instances with LMCache?

.. code-block:: console

   # Insert code here

* Launch a vLLM instance with LMCache and share the KV cache across multiple vLLM instances?

LMCache can share the KV cache across multiple vLLM instances. LMCache supports sharing KV using the ``lmcache.server`` module.
Here is a quick example:

.. code-block:: console

   # Start lmcache server
   $ lmcache_server localhost 65432
   
   # Then, start two vLLM instances with the LMCache config file

   $ wget https://raw.githubusercontent.com/LMCache/LMCache/refs/heads/dev/examples/example.yaml
   
   # start the first vLLM instance
   $ LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8000
   
   # start the second vLLM instance
   $ LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8001

