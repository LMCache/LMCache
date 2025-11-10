Benchmarking
============

This is a simple tutorial on how to deploy and benchmark LMCache. 

Workload Generator -- Long Doc QA:
----------------------------------

Long Doc QA (found in ``benchmarks/long_doc_qa/``) is a highly flexible traffic simulator that sends long context queries ("documents") to your serving engine.
Some configurable parameters include the number of tokens in the documents (default is 10000), the number of documents to send to the model (default is 20), the number of output tokens per request (default is 100), and the cache hit/miss ratio (e.g. 2:2 means a repeated 2 hit and 2 miss pattern through all the documents).
You can also choose the number of times to repeat prompts and the mode of repetition (random, tile, interleave).

LMCache provides a simple Long Doc QA Recommender that helps you deploy LMCache and generate the appropriate traffic through Long Doc QA.
It will also help you determine the tensor parallelism and the amount of CPU RAM to deploy LMCache with based on the specifications of your hardware.

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model <YOUR_MODEL_NAME>

Example:
---------

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model Qwen/Qwen3-8B

.. code-block:: text

    # this output is hardware specific, blindly copying it may not yield optimal results
    # please run the recommender script yourself
    1. vLLM Deployment: 
    -----------------

    PYTHONHASHSEED=0 \
    vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --load-format dummy


    2. LMCache Deployment: 
    --------------------

    PYTHONHASHSEED=0 \
    LMCACHE_MAX_LOCAL_CPU_SIZE=66 \
    vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --load-format dummy \
    --kv-transfer-config \
    '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'


    3. Multi-Round QA Workload Generation: 
    ----------------------------------------

    python benchmarks/long_doc_qa/long_doc_qa.py \
    --model Qwen/Qwen3-8B \
    --num-documents 46 \
    --document-length 10000 \
    --output-len 100 \
    --repeat-count 1 \
    --repeat-mode tile \
    --max-inflight-requests 4

Qwen 8B vLLM Metrics:
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Query round mean TTFT: 0.757s
    Query round time: 23.467s
    Query round prompt count: 46

Qwen 8B LMCache Metrics: 
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Query round mean TTFT: 0.185s
    Query round time: 13.789s
    Query round prompt count: 46

From this example, we can see a **75%** reduction in TTFT (0.757s → 0.185s), **41%** reduction in total inference time (23.467s → 13.789s) via offloading with **LMCache**.

.. note::
   The warmup round is the first time the model sees the documents. The query round is the second time the model sees the documents. Without offloading, even with KV Cache reuse, there is no improvement in TTFT nor throughput. With offloading, we can see significant performance improvements to the query round.
