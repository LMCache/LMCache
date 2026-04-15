L2 Serde (Serialization / Deserialization)
==========================================

LMCache supports a **per-adapter serde** that transforms KV cache data on
its way to and from an L2 adapter. Typical uses: quantization (shrink
storage footprint), compression, encryption. The serde runs
asynchronously in a background thread so it does not stall the L1 path.

.. contents::
   :local:
   :depth: 2


When to use serde
-----------------

- **Save L2 storage or bandwidth.** fp8 quantization halves byte volume
  vs. bf16 with minor accuracy loss — a good fit for disk / remote
  adapters.
- **Encrypt at rest.** Wrap the raw bytes with authenticated encryption
  before they land on disk.
- **Custom compression.** Anything lossless (lz4/zstd) or lossy
  (CacheGen-style) can be plugged in via the ``Serializer`` /
  ``Deserializer`` ABCs.

Serde is **opt-in per adapter**: one ``--l2-adapter`` may use fp8 while
another stores raw bytes. When omitted, the adapter behaves exactly as
before (no temp buffers, no extra allocations).


Data Flow
---------

**Store path** (L1 KV -> L2 bytes)::

    reserve_read(l1_keys)
        -> reserve_write(tmp_keys, byte layout)       # temp byte buffer
        -> serde.submit_serialize(l1_objs, tmp_objs)  # async
        -> [serialize_event_fd]                       # wake up
        -> finish_read(l1_keys)                       # release originals
        -> finish_write_and_reserve_read(tmp_keys)    # temp now readable
        -> L2 submit_store_task(tmp_objs)             # L2 reads temp
        -> [L2 store fd]
        -> finish_read(tmp_keys)                      # auto-deletes temp

**Prefetch path** (L2 bytes -> L1 KV)::

    reserve_write(keys_in_plan, KV layout)            # real KV buffer
        + reserve_write(tmp_keys, byte layout)        # temp byte buffer
        -> L2 submit_load_task(tmp_objs)              # L2 writes temp
        -> [L2 load fd]
        -> serde.submit_deserialize(tmp_objs, real_objs)   # async
        -> [deserialize_event_fd]                     # wake up
        -> finish_write(tmp_keys) + delete
        -> finish_write_and_reserve_read(keys_in_plan)

The serde processor registers two event fds — one for serialize, one
for deserialize — which the StoreController and PrefetchController
poll alongside the L2 adapter fds. Temp buffers are always allocated
through ``L1Manager`` so they count against L1 memory accounting.


Configuring serde on an L2 adapter
----------------------------------

Add a ``"serde"`` sub-dict to any ``--l2-adapter`` JSON spec. The ``type``
field selects a registered serde; remaining keys are forwarded to the
serde factory.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
        }'

.. list-table:: Built-in serde types
   :header-rows: 1
   :widths: 15 40 45

   * - ``type``
     - Description
     - Config fields
   * - ``fp8``
     - Quantize each element to 8-bit float; dequantize on load.
       Lossy but highly compressible.
     - ``fp8_dtype`` (default ``float8_e4m3fn``; also accepts
       ``float8_e5m2``), ``max_workers`` (thread pool size,
       default 1)


Writing a custom serde
----------------------

Implement the two sync ABCs (``Serializer``, ``Deserializer``) with your
transform logic, then register a factory keyed on a name you pick:

.. code-block:: python

    # my_project/my_serde.py
    from lmcache.v1.distributed.serde import (
        AsyncSerdeProcessor,
        Deserializer,
        Serializer,
        register_serde_factory,
    )

    class MySerializer(Serializer):
        def serialize(self, src, dst) -> int:
            # Write serialized bytes into dst; return bytes written.
            ...

        def estimate_serialized_size(self, layout_desc) -> int:
            # Upper bound on serialized byte size for this layout.
            ...

    class MyDeserializer(Deserializer):
        def deserialize(self, src, dst) -> None:
            # Read serialized bytes from src, write into dst (KV-shaped).
            ...

    def _create_mine(config: dict):
        return AsyncSerdeProcessor(MySerializer(), MyDeserializer())

    register_serde_factory("mine", _create_mine)

Reference it from your adapter config:

.. code-block:: json

    {"type": "fs", "base_path": "/data", "serde": {"type": "mine"}}

Only the sync ABC methods are required; ``AsyncSerdeProcessor`` takes
care of the thread pool, event fds, task ids, and completion signaling.


Implementation notes
--------------------

- **Temp buffer size.** The temp byte buffer is allocated at exactly
  ``serializer.estimate_serialized_size(layout)`` bytes. Your estimate
  must be an upper bound on the actual serialized output — include any
  safety margin directly in the estimate (e.g., the built-in fp8
  serializer returns ``1.5 * num_elements``).
- **Per-adapter independence.** A request can span adapters where some
  have serde and others do not. The prefetch controller waits for all
  L2 loads *and* all per-adapter deserializations before finalizing.
- **Failure handling.** If serialize fails, the temp buffer is cleaned
  up and no L2 store is submitted. If deserialize fails, the affected
  keys are treated as failed loads and cleaned up through the normal
  failed-key path.
- **Thread pool.** ``AsyncSerdeProcessor(max_workers=N)`` controls the
  pool size. Serde transforms that release the GIL (e.g., torch ops)
  benefit from ``N > 1``; pure-Python transforms do not.


Example
-------

An end-to-end script that starts an lmcache server with fp8 on a disk
adapter, runs vLLM, clears L1, and re-runs the same request to trigger
the L2 prefetch + fp8 deserialize path lives at
:file:`examples/serde_fp8/`. A standalone Python smoke test (no vLLM
required) that exercises the full L1 -> L2 -> L1 round-trip is at
:file:`examples/serde_fp8/smoke_test.py`.
