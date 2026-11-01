.. SPDX-License-Identifier: Apache-2.0

Engine-Driven Mode
==================

In ``engine_driven`` mode the vLLM workers gather and scatter the KV copies
themselves: the server hands over the requested chunks through a named
shared-memory pool, and the workers copy them into the paged KV on the device.
On NPU the default transfer mode (``auto``) resolves to ``lmcache_driven``, so
this mode must be selected explicitly via
``lmcache.mp.mp_transfer_mode: engine_driven``. It supports models with a
single KV-cache format only (standard GQA/MHA layouts, for example Qwen3);
multi-plane MLA/DSA models such as DeepSeek-V4 must use
:doc:`lmcache_driven`. **Mamba / GDN hybrids (Qwen3.5 / Qwen3.6 / Qwen3.8)
are not supported in this mode.** Use :doc:`lmcache_driven` and
:doc:`hybrid_mamba`.

.. warning::

   ``--shm-name`` and ``--no-l1-use-lazy`` are mandatory for this mode.
   Without a named shared-memory pool the server hands back ordinary
   (non-pinned) host memory, and every retrieve pays a dynamic pinning and
   extra copy on the worker, which degrades warm-hit performance severely.
   Always pair ``--shm-name`` with ``--no-l1-use-lazy`` so the pool is
   allocated eagerly at startup. A shared-memory pool of ``--l1-size-gb`` size
   is created under ``/dev/shm``, so the container must be started with a
   matching ``--shm-size``.

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

.. note::

   The shared-memory pool created by ``--shm-name`` lives under ``/dev/shm``;
   the ``--shm-size`` above must be at least as large as the server's
   ``--l1-size-gb``.

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
       --chunk-size 128 \
       --supported-transfer-mode engine_driven \
       --l1-size-gb 20 \
       --shm-name lmcache-mp-e2e --no-l1-use-lazy \
       --eviction-policy LRU &

Wait for readiness before starting vLLM:

.. code-block:: text

   LMCache INFO: Checking if shm capacity is larger than L1 request
   LMCache INFO: LMCache zmq cache server is running on localhost:5555
   INFO:     Uvicorn running on http://127.0.0.1:8080

Notes:

* ``--chunk-size 128`` matches the vLLM block size on Ascend and is a good
  default for single-KV-format models; unlike DeepSeek-V4 (see
  :doc:`lmcache_driven`), there is no fixed chunk-size requirement here.
* ``--l1-size-gb`` sizes the shared-memory pool; size it for the expected
  working set (number of distinct prefixes x prefix tokens x KV bytes per
  token) so that it fits comfortably.

4.2 Start vLLM
~~~~~~~~~~~~~~

Example: Qwen3-32B, TP8 on 8 NPUs.

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

   vllm serve /workspace/models/Qwen3-32B \
       --served-model-name qwen3 \
       --max-model-len 40960 \
       --max-num-batched-tokens 8192 \
       --gpu-memory-utilization 0.8 \
       --max-num-seqs 32 \
       --tensor-parallel-size 8 \
       --no-enable-prefix-caching \
       --port 8900 \
       --kv-transfer-config '{
         "kv_connector": "LMCacheMPConnector",
         "kv_role": "kv_both",
         "kv_connector_extra_config": {
           "lmcache.mp.mp_transfer_mode": "engine_driven"
         }
       }'

The connector reaches the server at ``localhost:5555`` by default; only the
transfer mode needs to be set explicitly. A successful start shows, for every
worker:

.. code-block:: text

   LMCache INFO: Creating transfer context (device_type=npu, mode=engine_driven)
   LMCache INFO: Engine KV Format: EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
   LMCache INFO: Registered non-GPU context for instance ... (world_size=8)
   LMCache INFO: Using shm non-GPU transfer strategy
   LMCache INFO: Creating EngineDrivenContextShm (shm_name=...)

``Creating EngineDrivenContextShm`` confirms the workers attached the
shared-memory pool; if this line is absent, re-check the server's
``--shm-name`` / ``--no-l1-use-lazy`` flags.

.. note::

   Leave NPU memory headroom for transient transfer buffers inside the worker:
   keep ``--gpu-memory-utilization`` at or below ``0.8`` instead of the usual
   ``0.9``, because the retrieve path allocates temporary device buffers on top
   of vLLM's profiled budget.

5. Verification
---------------

Send the same long prompt twice with ``temperature: 0``:

.. code-block:: bash

   curl -s http://localhost:8900/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"qwen3","messages":[{"role":"user","content":"<long prompt>"}],"max_tokens":100,"temperature":0}'

* The server log shows ``Stored ... tokens`` for the first request and
  ``Retrieved ... tokens`` for the second.
* The vLLM request log shows ``External prefix cache hit rate`` growing above
  zero.
* The two responses are identical, which confirms the KV round-trip is correct
  for this single-format model.
