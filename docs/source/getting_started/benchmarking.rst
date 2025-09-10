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

First set your ``HF_TOKEN`` environment variable with access to the model you want to benchmark. Then run the recommendation script: 

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model <YOUR_MODEL_NAME>

Example #1:
-----------

.. code-block:: bash

    # default is meta-llama/Meta-Llama-3.1-8B-Instruct
    python benchmarks/long_doc_qa/long_doc_qa_recommender.py

.. code-block:: text

    # this output is hardware specific, blindly copying it may not yield optimal results
    # please run the recommender script yourself
    1. vLLM Deployment: 
    -----------------

    PYTHONHASHSEED=0 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --load-format dummy


    2. LMCache Deployment: 
    --------------------

    PYTHONHASHSEED=0 \
    LMCACHE_MAX_LOCAL_CPU_SIZE=66 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --load-format dummy \
    --kv-transfer-config \
    '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'


    3. Multi-Round QA Workload Generation: 
    ----------------------------------------

    python benchmarks/long_doc_qa/long_doc_qa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-documents 51 \
    --document-length 10000 \
    --output-len 100 \
    --repeat-count 1 \
    --repeat-mode tile \
    --max-inflight-requests 4

Llama 8B vLLM Metrics:
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Warmup round mean TTFT: 0.751s
    Warmup round time: 24.915s
    Warmup round prompt count: 51
    Query round mean TTFT: 0.753s
    Query round time: 24.628s
    Query round prompt count: 51

Llama 8B LMCache Metrics: 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Warmup round mean TTFT: 0.832s
    Warmup round time: 26.027s
    Warmup round prompt count: 51
    Query round mean TTFT: 0.214s
    Query round time: 14.564s
    Query round prompt count: 51

The warmup round is the first time the model sees the documents. The query round is the second time the model sees the documents. Without offloading, even with KV Cache reuse, there is no improvement in TTFT nor throughput. With offloading, we can see significant performance improvements to the query round. 

Example #2:
-----------

.. code-block:: bash

    python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model meta-llama/Llama-3.1-70B-Instruct

.. code-block:: text

    1. vLLM Deployment: 
    -----------------

    PYTHONHASHSEED=0 \
    vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --load-format dummy


    2. LMCache Deployment: 
    --------------------

    PYTHONHASHSEED=0 \
    LMCACHE_MAX_LOCAL_CPU_SIZE=40 \
    vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --load-format dummy \
    --kv-transfer-config \
    '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'


    3. Multi-Round QA Workload Generation: 
    ----------------------------------------

    python benchmarks/long_doc_qa/long_doc_qa.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --num-documents 50 \
    --document-length 10000 \
    --output-len 100 \
    --repeat-count 1 \
    --repeat-mode tile \
    --max-inflight-requests 4

Llama 70B vLLM Metrics:
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Warmup round mean TTFT: 1.797s
    Warmup round time: 54.903s
    Warmup round prompt count: 50
    Query round mean TTFT: 1.798s
    Query round time: 54.974s
    Query round prompt count: 50

Llama 70B LMCache Metrics: 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    === BENCHMARK RESULTS ===
    Warmup round mean TTFT: 1.881s
    Warmup round time: 56.673s
    Warmup round prompt count: 50
    Query round mean TTFT: 0.174s
    Query round time: 26.223s
    Query round prompt count: 50
