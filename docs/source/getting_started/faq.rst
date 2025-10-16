FAQ
===

What are the KV cache sizes for popular models? And why is LMCache important?
-----------------------------------------------------------------------------

You can calculate KV cache sizes using our :doc:`KV cache calculator <kv_cache_calculator>`. We also provide a reference table below with KV cache information for some popular models.

As shown in the table, after loading Qwen/Qwen3-32B for example, there is only enough space in the spare GPU RAM to hold 275,760 tokens for KV caches. This supports only 6.73 concurrent users if each prompt is 40,960 tokens long. Once this capacity is exceeded, the KV cache must be evicted, and when the same user returns, their request needs to be re-prefilled, which takes significantly longer.

**LMCache is designed to extend this virtual memory capacity**, enabling you to store more KV caches and avoid costly re-prefilling operations.

**KV Cache Sizes for Popular Models**

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15 15

   * - Model
     - KV Cache Size per 1000 tokens
     - Spare GPU RAM for KV cache
     - Context length
     - Number of full-length prompts that can be stored in GPU
   * - Qwen/Qwen3-8B
     - 0.1373 GB
     - 50.32 GB (or 366,400 tokens)
     - 40,960 tokens
     - 8.95x
   * - Qwen/Qwen3-32B (tp=2 on H100)
     - 0.2441 GB
     - 33.66 GB × 2 (or 275,760 tokens)
     - 40,960 tokens
     - 6.73x
   * - meta-llama/Llama-3.1-70B (tp=4 on H100)
     - 0.3052 GB
     - 32.06 GB × 4 (or 420,208 tokens)
     - 131,072 tokens
     - 3.21x

.. note::
   You may also find this `VRAM Calculator <https://apxml.com/tools/vram-calculator>`_ useful for calculating the estimated spare GPU RAM for different models and configurations. 