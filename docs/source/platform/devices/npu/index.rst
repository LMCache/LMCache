.. SPDX-License-Identifier: Apache-2.0

Ascend NPU
==========

The ``npu`` backend runs LMCache on Ascend NPUs with
`vLLM-Ascend <https://github.com/vllm-project/vllm-ascend>`_ as the serving
engine. The in-tree device package
(``lmcache/v1/platform/devices/npu``) implements the community ``DeviceSpec``
and ``DeviceOps`` interfaces, the per-layer IPC wrapper, event IPC, cache
context, and host-memory pinning. Compiled NPU kernels and transfer
optimizations live in the external
`LMCache-Ascend <https://github.com/LMCache/LMCache-Ascend>`_ plugin
(``lmcache_ascend``); the in-tree backend binds them through
``lmcache_ascend.ops``.

NPU supports both multiprocess (MP) transfer modes and, unlike the
engine-driven backends, defaults to **LMCache-driven** transfer. That default
is required for multi-plane MLA/DSA models such as DeepSeek-V4.

Prerequisites
-------------

Validated on Atlas A2/A3 (Ascend 910B). Atlas 300I Duo is not supported.

- CANN Toolkit >= 8.5.0 and Ascend driver >= 25.5.
- Python >= 3.10 (LMCache-Ascend supports 3.10-3.12).
- PyTorch >= 2.8.0 with the ``torch_npu`` package.
- vLLM and vLLM-Ascend >= 0.23.0.
- Containers started with the Ascend driver and the ``/dev/davinci*``
  devices passed through. Follow the official
  `vLLM-Ascend tutorials <https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/index.html>`_
  for base environment preparation.

Installation
------------

Install LMCache and LMCache-Ascend from source in the same Python
environment:

.. code-block:: bash

   git clone https://github.com/LMCache/LMCache.git
   cd LMCache
   pip install -v --no-build-isolation -e .
   cd ..

   git clone --recurse-submodules \
       https://github.com/LMCache/LMCache-Ascend.git
   cd LMCache-Ascend
   pip install -v --no-build-isolation -e .

The ``third_party/hcomm`` submodule must match the container's CANN version.
Check the installed CANN version, switch the submodule to the matching tag
from `the hcomm repository <https://gitcode.com/cann/hcomm>`_, then build:

.. code-block:: bash

   ls /usr/local/Ascend        # e.g. 8.5.0 means CANN 8.5
   cd third_party/hcomm
   git fetch --tags
   git checkout v8.5.0         # tag matching the CANN version
   cd ../..
   pip install -v --no-build-isolation -e .

Use matching revisions of the two repositories: each published
LMCache-Ascend release pairs with the LMCache release of the same version
(``v0.5.6`` with ``v0.5.6``), and ``main`` tracks ``dev``. MP mode replaces
the in-process connector, so do not install the LMCache release used by
``LMCacheAscendConnectorV1Dynamic`` (LMCache 0.4.5) in the same environment.

Verify the installation:

.. code-block:: bash

   python -c "import lmcache_ascend"
   lmcache server --help | grep supported-transfer-mode

LMCache selects ``npu`` automatically when ``torch.npu`` reports an
available device. To pin the backend, set:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=npu

With the backend pinned, startup fails when the environment has no
``torch.npu`` module instead of falling back to the CPU stub.

Running vLLM-Ascend with LMCache
--------------------------------

Start the LMCache server:

.. code-block:: bash

   lmcache server --chunk-size 128 --l1-size-gb 50 \
       --eviction-policy LRU

The server defaults to ``--supported-transfer-mode lmcache_driven``, which
is also the NPU worker default, so no transfer-mode flag is needed.
``--chunk-size`` is a minimum: the connector rounds it up to a multiple of
the vLLM cache-group alignment. 128 matches the Ascend block size for
single-KV-format models, while DeepSeek-V4 requires 4096 because its packed
MLA/DSA layout is designed for 4096-token chunks. Size ``--l1-size-gb`` so
that the whole working set fits; otherwise LRU eviction turns hits back into
misses.

Start vLLM-Ascend with ``LMCacheMPConnector``:

.. code-block:: bash

   vllm serve <model> \
       ... \
       --kv-transfer-config '{
         "kv_connector": "LMCacheMPConnector",
         "kv_role": "kv_both"
       }'

Leave ``lmcache.mp.mp_transfer_mode`` unset (or set it to ``auto``): on NPU
it resolves to ``lmcache_driven``. Each worker logs the resolved mode:

.. code-block:: text

   LMCache INFO: Creating transfer context (device_type=npu, mode=lmcache_driven)

Engine-driven mode
------------------

To use engine-driven transfer instead, select it on both sides. On the
server, also create a named shared-memory pool:

.. code-block:: bash

   lmcache server --chunk-size 128 --l1-size-gb 20 \
       --supported-transfer-mode engine_driven \
       --shm-name lmcache-mp-e2e --no-l1-use-lazy

On the worker, override the transfer mode in the connector configuration:

.. code-block:: json

   "kv_connector_extra_config": {
     "lmcache.mp.mp_transfer_mode": "engine_driven"
   }

``--shm-name`` creates a pool under ``/dev/shm``, so the container's
``--shm-size`` must be at least ``--l1-size-gb``. Keep
``--gpu-memory-utilization`` at or below 0.8 because the retrieve path
allocates transient device buffers on top of vLLM's profiled budget.

Supported KV cache formats
--------------------------

* **LMCache-driven:** standard GQA/MHA layouts, MLA/DSA multi-plane layouts
  (for example DeepSeek-V4 ``(latent, rope, ...)``), and hybrid Mamba models
  (for example Qwen3.5) with ``--mamba-cache-mode align``. The IPC wrapper
  moves each per-layer K/V pair or multi-plane tuple through ``torch_npu``
  storage IPC.
* **Engine-driven:** single KV-cache format models with standard GQA/MHA
  layouts only.

In MP mode, CacheBlend and CacheGen are not supported, and
``--mamba-cache-mode all`` is not supported.

Validation
----------

Run the backend tests on a host with an Ascend NPU:

.. code-block:: bash

   python -m pytest -q -rs tests/v1/platform/devices/npu

Tests that need an NPU skip when no device can be opened; skipped cases are
not hardware validation.

For an end-to-end check, send the same long prompt twice with
``temperature: 0``. The first request stores KV chunks and the second
retrieves them: the server log shows ``Stored ... tokens`` and
``Retrieved ... tokens``, and vLLM's ``External prefix cache hit rate`` rises
above zero.

Further documentation
---------------------

The LMCache-Ascend repository contains the full deployment guides:

* `LMCache-driven mode <https://github.com/LMCache/LMCache-Ascend/blob/main/docs/mp_mode/lmcache_driven.md>`_
* `Engine-driven mode <https://github.com/LMCache/LMCache-Ascend/blob/main/docs/mp_mode/engine_driven.md>`_
* `Hybrid Mamba models <https://github.com/LMCache/LMCache-Ascend/blob/main/docs/mp_mode/qwen3_5.md>`_
