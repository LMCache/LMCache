CacheGen
========

Multiprocess mode can store L2 objects with CacheGen compression by
configuring an L2 adapter serde:

.. code-block:: json

   {
     "type": "fs",
     "base_path": "/tmp/lmcache",
     "serde": {
       "type": "cachegen",
       "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
       "chunk_size": 256,
       "dtype": "bfloat16",
       "num_heads": 8,
       "head_size": 128
     }
   }

Required serde fields:

``model_name``
   Model identifier used to select CacheGen quantization settings.

``chunk_size``
   Maximum number of tokens in each cached KV object.

``dtype``
   Destination KV dtype on load, for example ``bfloat16``.

``num_heads``
   Number of KV heads in the MP KV layout.

``head_size``
   Per-head KV dimension. ``num_heads * head_size`` must equal the MP
   layout's hidden dimension.

Optional serde fields:

``max_workers``
   Thread-pool size for asynchronous encode/decode work. The default is ``1``.

CacheGen uses LMCache's existing CacheGen kernels, so encode/decode requires a
backend where those kernels are available. CPU-only environments can parse the
config but cannot run CacheGen encode/decode.

CacheGen serialized payloads use LMCache's existing CacheGen bytestream format.
Only enable this serde for L2 storage you trust; do not decode CacheGen objects
written by untrusted users or systems.

On load, LMCache uses the exact stored byte length when the L2 adapter can
report it; the filesystem adapter reports this from the object file size. This
matters for CacheGen because the encoded payload is variable-sized. Adapters
that cannot report exact sizes fall back to the serializer's conservative
estimate; the destination KV object still uses the original KV shape and
``dtype``.

For algorithm background and legacy in-process configuration, see
:doc:`/kv_cache_optimizations/compression/cachegen`.
