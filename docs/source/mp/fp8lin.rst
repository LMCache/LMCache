fp8lin (lossless)
=================

The ``fp8lin`` serde stores each KV chunk as E4M3 codes, entropy coded on the GPU
against a fitted linear prediction from the token's other heads and its previous
layer. It is **lossless for FP8 KV**: decoding returns the stored bytes exactly.
Typical size is 4.2–4.8 bits per value, **1.7–1.9× smaller than FP8** (Qwen3
1.7B–14B).

Use it where L2 capacity or bandwidth is the constraint and FP8 KV must stay exact:
more prefix cache per byte of disk, or less traffic to a remote L2.

Configuring
-----------

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {"type": "fp8lin", "coder_path": "/models/qwen3-8b-fp8lin",
                      "chunk_size": 256}
        }'

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``coder_path``
     - Directory with the fitted coder (``meta.json`` and one ``layer_XXX.npz``
       per layer). Required.
   * - ``chunk_size``
     - LMCache chunk size in tokens (default 256). Chunk start positions are
       searched on multiples of it.
   * - ``device``
     - CUDA device for coding (default: the current device).

A coder is fitted once per model and KV convention (bf16 KV rounded to E4M3, or an
FP8 KV cache) on a few hundred documents; see ``lmcache/v1/fp8lin/README.md``.

Behavior
--------

- **Variable-size objects.** ``serialize`` returns the bytes it wrote and the async
  processor narrows the buffer to them, so L2 stores exactly the coded size. On
  load, the filesystem adapter accepts a short read into a serialized buffer.
  Other L2 adapters still require full-size reads and do not support ``fp8lin``
  yet.
- **Exactness.** Each blob carries a checksum; a corrupt or mismatching blob fails
  its load (a cache miss), never returns wrong KV.
- **Same GPU type** for encode and decode.
- **bf16 / fp16 KV** is rounded to E4M3 at the coder's per-layer scales before
  coding (lossy, like ``fp8``).
- **FP8 KV caches** (``--kv-cache-dtype fp8``) are coded from their stored bytes.
  Measured with vLLM and Qwen3-1.7B: 33,353 bytes per token in L2 instead of
  57,344, and the response after an L2 restore matches the cold response exactly.
  The ``fp8`` serde does not support FP8 KV caches: it casts the stored uint8 codes
  numerically.

A filesystem round-trip test without vLLM is at
:file:`tests/v1/distributed/serde/test_fp8lin.py`; an end-to-end example with vLLM
is at :file:`examples/serde/fp8lin/`.
