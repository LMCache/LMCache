.. SPDX-License-Identifier: Apache-2.0

LMCache-Driven Mode
===================

In ``lmcache_driven`` mode the LMCache server pulls and pushes KV cache
through device IPC handles: the engine submits a lookup/retrieve request, and
the server performs the data movement on the engine's behalf. This mode
supports standard GQA/MHA layouts as well as vLLM **MLA/DSA multi-plane**
packed KV (``(latent, scale)`` layouts) and is the required mode for
DeepSeek-V4.

Configuration constraints
-------------------------

This section lists the configuration constraints for each model. Refer to
:doc:`../../../recipes/index` for a complete list of recipes.

**Model / model family:**

* **DeepSeek-V4.** ``--chunk-size 4096`` is a DeepSeek-V4 requirement, not a
  general one. DeepSeek-V4's packed MLA/DSA KV layout is designed for
  4096-token chunks, and the connector expects this size for that model
  family.
* **Dense GQA/MHA (for example Qwen).** Models with a single KV-cache format
  do not need 4096 and can use a different chunk size (for example 128, which
  matches the vLLM block size).
* **Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8).** Served
  in this mode. Use the :doc:`../../../recipes/qwen3_5` recipe for
  ``--mamba-cache-mode align``, ``--enable-prefix-caching``,
  ``--separate-object-groups``, and ``LMCacheMPConnector`` with
  ``kv_role: kv_both``. CacheBlend and CacheGen stay unsupported.
  ``--mamba-cache-mode all`` is not supported.

  * Leave ``--block-size`` unset. Set ``--chunk-size`` and
    ``--max-num-batched-tokens`` from the table in :doc:`hybrid_mamba` for
    your model and vLLM-Ascend version. Do not copy the chunk size from the
    CUDA recipe.
  * If you set ``--block-size``, change the cache dtype, or enable speculative
    decoding, that table does not apply. ``--chunk-size`` is the block size
    vLLM prints at startup (``Setting attention block size to ...``), and
    ``--max-num-batched-tokens`` is ``2N - 1``.
  * ``--l1-size-gb`` has no fixed value. Every rank stores a full copy of the
    pool.

1. Environment Preparation
--------------------------

.. note::

   **Hardware validation scope:** this guide has been verified on Ascend 910B
   (A2 / A3).

For base environment preparation (NPU driver, CANN toolkit, container
images), follow the official `vLLM-Ascend tutorials
<https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/index.html>`_.

Create the working directory on the host:

.. code-block:: bash

   mkdir -p /workspace

.. note::

   This directory is mounted into the container for storing source code,
   configs, and logs.

2. Docker Container Startup
---------------------------

Launch the Ascend NPU vLLM container:

.. code-block:: bash

   #!/bin/bash
   export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a3
   docker run \
       --name vllm-ascend \
       --shm-size=512g \
       --net=host \
       --privileged=true \
       --device /dev/davinci0 \
       --device /dev/davinci1 \
       --device /dev/davinci2 \
       --device /dev/davinci3 \
       --device /dev/davinci4 \
       --device /dev/davinci5 \
       --device /dev/davinci6 \
       --device /dev/davinci7 \
       --device /dev/davinci_manager \
       --device /dev/devmm_svm \
       --device /dev/hisi_hdc \
       -v /usr/local/dcmi:/usr/local/dcmi \
       -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
       -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
       -v /etc/ascend_install.info:/etc/ascend_install.info \
       -v /etc/hccn.conf:/etc/hccn.conf \
       -v /workspace:/workspace \
       -v /home:/home \
       -v /root/.cache:/root/.cache \
       -it $IMAGE bash

3. Install from Source (Inside the Container)
----------------------------------------------

3.1 Install LMCache
~~~~~~~~~~~~~~~~~~~

The LMCache main branch (``dev``) provides the MP transfer architecture (the
``lmcache server`` CLI and the ``LMCacheMPConnector``):

.. code-block:: bash

   cd /workspace
   git clone https://github.com/LMCache/LMCache.git
   cd LMCache
   pip install -v --no-build-isolation -e .
   cd ..

.. warning::

   Do not mix this checkout with the LMCache release used by the in-process
   mode (for example ``v0.4.x`` installed for
   ``LMCacheAscendConnectorV1Dynamic``): both are installed editable and only
   one can own the ``lmcache`` package in a given environment.

3.2 Install LMCache-Ascend
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /workspace
   git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
   cd LMCache-Ascend

.. warning::

   ``third_party/hcomm`` must match the container's CANN version.

1. Check the container's CANN version:

   .. code-block:: bash

      ls /usr/local/Ascend

   For example, ``8.5.0`` (or ``0.8.5``) means CANN 8.5; ``9.0.0`` (or
   ``0.9.0``) means CANN 9.0.

2. List available hcomm tags from the `hcomm repository
   <https://gitcode.com/cann/hcomm>`_ and check out the tag that matches the
   CANN version **before** running ``pip install``:

   .. code-block:: bash

      cd third_party/hcomm
      git fetch --tags
      git checkout v8.5.0   # CANN 8.5.x; use the matching tag
      cd ../..

3. If you switch (or roll back) the submodule, always rebuild:

   .. code-block:: bash

      pip install -v --no-build-isolation -e .

Install the package:

.. code-block:: bash

   pip install -v --no-build-isolation -e .

3.3 Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip show lmcache lmcache-ascend                       # both installed editable
   python3 -c "import lmcache_ascend"                    # imports cleanly on the NPU host
   lmcache server --help | grep supported-transfer-mode  # lmcache_driven / engine_driven / auto

4. Service Startup
------------------

4.1 Start the LMCache Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   lmcache server \
       --chunk-size 4096 \
       --l1-size-gb 50 \
       --eviction-policy LRU &

Wait for readiness before starting vLLM:

.. code-block:: text

   LMCache INFO: LMCache zmq cache server is running on localhost:5555
   INFO:     Uvicorn running on http://127.0.0.1:8080

Notes:

* The server's transfer mode defaults to ``lmcache_driven``, so
  ``--supported-transfer-mode`` is not needed here (unlike
  :doc:`engine_driven`, which must set it).
* The server binds the ZMQ endpoint on ``localhost:5555`` by default
  (``--host`` / ``--port`` to change) and an HTTP health/ops endpoint on port
  ``8080`` on the same host; make sure both are free.
* Size ``--l1-size-gb`` so that the whole working set (number of distinct
  prefixes x prefix tokens x KV bytes per token) fits comfortably; if the
  working set exceeds L1, LRU eviction turns hits back into misses and
  requests fall back to full prefill.
* The server and the engines share failure semantics: restart the server and
  the engines together, and never restart the server while requests are in
  flight.

4.2 Start vLLM
~~~~~~~~~~~~~~

Example: DeepSeek-V4-Flash, TP8/EP on 8 NPUs.

.. code-block:: bash

   #!/bin/bash
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
   export HCCL_BUFFSIZE=1024
   export TASK_QUEUE_ENABLE=1
   export HCCL_OP_EXPANSION_MODE="AIV"

   vllm serve /workspace/models/DeepSeek-V4-Flash-0731-w8a8 \
       --max-model-len 133120 \
       --max-num-batched-tokens 8192 \
       --served-model-name dsv4 \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 \
       --data-parallel-size 1 \
       --tensor-parallel-size 8 \
       --enable-expert-parallel \
       --tokenizer-mode deepseek_v4 \
       --tool-call-parser deepseek_v4 \
       --enable-auto-tool-choice \
       --reasoning-parser deepseek_v4 \
       --no-enable-prefix-caching \
       --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
       --quantization ascend \
       --port 8900 \
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       --additional-config '
       {"ascend_compilation_config":{
           "enable_npugraph_ex":true,
           "enable_static_kernel":false
           },
       "enable_cpu_binding": false,
       "enable_dsa_cp": false,
       "multistream_overlap_shared_expert": true}' \
       --kv-transfer-config '{
         "kv_connector": "LMCacheMPConnector",
         "kv_role": "kv_both"
       }'

The connector reaches the server at ``localhost:5555`` by default, and on NPU
the transfer mode (``auto``) resolves to ``lmcache_driven``. A successful
connector start shows this line for every worker:

.. code-block:: text

   LMCache INFO: Creating transfer context (device_type=npu, mode=lmcache_driven)

5. Verification
---------------

Send the same long prompt twice with ``temperature: 0``. The first request
prefills the prompt and stores the KV cache; the second request hits the
external cache:

.. code-block:: bash

   curl -s http://localhost:8900/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"dsv4","messages":[{"role":"user","content":"<long prompt>"}],"max_tokens":100,"temperature":0}'

Expected evidence:

* The server log shows ``Stored ... tokens`` for the first request and
  ``Retrieved ... tokens`` for the second request.
* The vLLM request log shows ``External prefix cache hit rate`` growing above
  zero.
* The two responses are coherent completions; with the quantized KV round-trip
  the warm continuation may drift by a few tokens from the cold one, which is
  expected.

6. Tips
-------

* ``--no-enable-prefix-caching`` is used in the reference command so that
  external-cache hits are attributable to LMCache; internal prefix caching can
  be enabled independently if desired.
* The connector reports lookup/load progress asynchronously: requests waiting
  for an external load appear as ``Deferred`` in the vLLM engine log line
  instead of blocking the engine.
* Do not resize or delete the shared-memory pool while any engine is attached.
* **To reveal LMCache advantages under load:** increase request concurrency,
  or keep ``--no-enable-prefix-caching`` to test LMCache IO efficiency in
  isolation.
