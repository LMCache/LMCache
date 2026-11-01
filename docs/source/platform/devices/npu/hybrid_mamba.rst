.. SPDX-License-Identifier: Apache-2.0

Hybrid Mamba Models (Qwen3.5 / Qwen3.6 / Qwen3.8)
=================================================

Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8, architecture
``Qwen3_5ForConditionalGeneration``) interleave Mamba / Gated-DeltaNet layers
with full attention. Serve them with MP mode in ``lmcache_driven``.
``engine_driven`` does not support this family.

Flags that match the LMCache recipe are not repeated here. Follow the
:doc:`../../../recipes/qwen3_5` recipe for:

* ``--mamba-cache-mode align`` (required; ``all`` is not supported)
* ``--enable-prefix-caching``
* ``--separate-object-groups`` on ``lmcache server``
* ``--chunk-size`` equal to the attention block size ``N`` defined below
* ``--max-num-batched-tokens`` at least ``N``, with ``2N - 1`` as the default
* ``LMCacheMPConnector`` with ``kv_role: kv_both``
* CacheBlend and CacheGen stay unsupported
* cached and fresh generations match at score level, not token level

The dense Qwen example in :doc:`index` (``--chunk-size 128``,
``--no-enable-prefix-caching``) is a different model family. Do not use it
here.

What differs on Ascend
----------------------

Leave ``--block-size`` unset. ``--chunk-size`` is ``N`` from the table for
your model and vLLM-Ascend version. ``--max-num-batched-tokens`` is
``2N - 1``. v0.23 and v0.30 are different rows; do not mix them. The CUDA
recipe table is not a row here.

The table is the default configuration: no ``--block-size``, the model's own
cache dtypes, no speculative decoding. Tensor parallel size does not change
it.

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Model
     - vLLM-Ascend
     - ``--chunk-size``
     - ``--max-num-batched-tokens``
   * - Qwen3.8-27B
     - v0.30
     - 896
     - 1791
   * - Qwen3.8-27B
     - v0.23
     - 1536
     - 3071
   * - Qwen3.5-4B
     - v0.30
     - 640
     - 1279
   * - Qwen3.5-4B
     - v0.23
     - 1024
     - 2047

Qwen3.6-27B and Qwen3.5-0.8B are not in this table. For those, and for any
run that sets ``--block-size``, ``--kv-cache-dtype``,
``--mamba-ssm-cache-dtype``, or speculative decoding, take ``N`` from the
startup line ``Setting attention block size to ...``. A chunk that is not a
multiple of that ``N`` is rejected at connector startup.

``--l1-size-gb`` has no fixed value. Every rank stores a full copy of the
pool.

On NPU, an unset transfer mode resolves to ``lmcache_driven``, and the server
address defaults to ``localhost:5555``. Pass ``lmcache.mp.host``,
``lmcache.mp.port``, and ``lmcache.mp.mp_transfer_mode`` only when those
defaults are wrong.

Example
-------

Qwen3.8-27B on vLLM-Ascend v0.30, the first row of the table. Use the row for
the version you run.

.. code-block:: bash

   lmcache server \
       --chunk-size 896 \
       --separate-object-groups \
       --eviction-policy LRU

.. code-block:: bash

   vllm serve /path/to/Qwen3.8-27B \
       --tensor-parallel-size 2 \
       --enable-prefix-caching \
       --mamba-cache-mode align \
       --max-num-batched-tokens 1791 \
       --kv-transfer-config '{
         "kv_connector": "LMCacheMPConnector",
         "kv_role": "kv_both"
       }'
