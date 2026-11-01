.. SPDX-License-Identifier: Apache-2.0

MP Mode on Ascend NPU
=====================

MP mode is the multiprocess deployment architecture of LMCache: the KV cache
is owned by a standalone ``lmcache server`` process, and vLLM attaches to it
through the ``LMCacheMPConnector``. The server owns L1/L2 storage, eviction,
and lookup; the engine talks to it over the MP protocol (ZMQ control plane +
shared-memory / device-handle data plane). Compared with running LMCache inside
the vLLM process, this keeps cache work out of the engine loop and allows one
cache to be shared by several engines.

The data plane has two transfer modes, selected with the
``lmcache.mp.mp_transfer_mode`` key in ``kv_connector_extra_config``:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - ``lmcache_driven``
     - ``engine_driven``
   * - Data path
     - The LMCache server pulls/pushes KV through device IPC handles.
     - The vLLM workers gather/scatter the KV copies themselves, zero-copy
       from a named shared-memory pool.
   * - KV cache formats
     - Standard GQA/MHA, MLA/DSA multi-plane layouts (including DeepSeek-V4
       ``(latent, scale)``), and hybrid Mamba models (for example Qwen3.5,
       Qwen3.6, and Qwen3.8; see :doc:`hybrid_mamba`).
     - Single KV-cache format models only (standard GQA/MHA, for example
       Qwen3); not Mamba/GDN or MLA/DSA.
   * - Server requirements
     - ``--chunk-size 4096`` for DeepSeek-V4 (see :doc:`lmcache_driven`).
     - ``--shm-name <name> --no-l1-use-lazy`` are mandatory (see
       :doc:`engine_driven`).
   * - Deployment guide
     - :doc:`lmcache_driven`
     - :doc:`engine_driven`

On NPU, leaving ``lmcache.mp.mp_transfer_mode`` unset (or ``auto``) resolves
to ``lmcache_driven``, so DeepSeek-V4 needs no extra setting; to use
``engine_driven`` you must set it explicitly.

Supported models and limitations
--------------------------------

**Supported in MP mode (both transfer modes, where the format applies):**

* Standard dense GQA/MHA models (for example Qwen3).
* Multi-plane MLA/DSA packed KV layouts (for example DeepSeek-V4) when using
  **``lmcache_driven``**.
* Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8, architecture
  ``Qwen3_5ForConditionalGeneration``) when using **``lmcache_driven``**. The
  flags that match the LMCache recipe, and the Ascend differences, are in
  :doc:`hybrid_mamba`.

In the table above, "MLA/DSA multi-plane layouts" refers to attention KV
layouts, not SSM/Mamba hybrids.

**Not supported in MP mode (current release):**

* Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8) in
  **``engine_driven``**. Use ``lmcache_driven``.
* ``--mamba-cache-mode all``. These models require ``align``.

Further reading
---------------

* :doc:`lmcache_driven` -- full ``lmcache_driven`` deployment guide:
  environment preparation, container startup, source installation,
  server/vLLM startup, and verification (DeepSeek-V4 example).
* :doc:`engine_driven` -- deploy and run ``engine_driven`` (Qwen3 example).
* :doc:`hybrid_mamba` -- hybrid Mamba models on Ascend: the flags that match
  the :doc:`../../../recipes/qwen3_5` recipe, plus the Ascend differences.

Relationship to the LMCache dev branch
--------------------------------------

The MP architecture (the ``lmcache server`` CLI, the ``LMCacheMPConnector``,
and the transfer-mode layer) is developed on the LMCache ``dev`` branch. The
matching Ascend backends (NPU platform routing, block-transfer ops, hccl/hixl
bindings) live in the LMCache-Ascend repository. Install both from the
official repositories only -- see :doc:`lmcache_driven` for the exact
procedure. Do not mix this checkout with the LMCache release used by the
legacy in-process connector (``LMCacheAscendConnectorV1Dynamic``) in one
environment.
