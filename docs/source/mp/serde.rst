L2 Serde (Serialization / Deserialization)
==========================================

LMCache supports a **per-adapter serde** that transforms KV cache data on
its way to and from an L2 adapter. Typical uses: quantization (shrink
storage footprint), compression, encryption. The serde runs
asynchronously in a background thread so it does not stall the L1 path.

.. contents::
   :local:
   :depth: 2


Design in one paragraph
-----------------------

Serde is an **in-CPU pipe in front of each L2 adapter** — every byte
between L1 and the adapter flows through it. The choice is
**per-adapter and all-or-nothing**: an adapter either pipes everything
through serde or has no pipe at all. Internally, a serde-enabled
adapter is wrapped by ``SerdeL2AdapterWrapper``, which presents a
normal ``L2AdapterInterface`` to the controllers while transparently
serializing / deserializing via an injected ``SerdeProcessor``. The
pipe needs a temp buffer to hold the serialized bytes; that buffer is
explicitly allocated through ``L1Manager`` so the extra memory is
visible in L1 accounting rather than hidden. From the caller's point
of view, nothing about the L2 interface changes — lookup, load, store,
and eviction all behave the same whether or not serde is attached.


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
another stores raw bytes. When omitted, the adapter is instantiated
directly (no wrapper, no temp buffers, no extra threads).


Data flow
---------

**Store path** (L1 KV → L2 bytes), driven by the wrapper's internal
thread::

    caller: submit_store_task(l1_keys, l1_objs)
        wrapper: reserve_write(tmp_keys, byte layout)          # temp buffer
        wrapper: serde.submit_serialize(l1_objs, tmp_objs)     # async
        -> [serialize_event_fd]
        wrapper: finish_write_and_reserve_read(tmp_keys)       # temp → readable
        wrapper: inner.submit_store_task(l1_keys, tmp_objs)
        -> [inner store_event_fd]
        wrapper: finish_read(tmp_keys)                         # auto-deletes
    caller: pop_completed_store_tasks() → {id: True}

**Prefetch path** (L2 bytes → L1 KV)::

    caller: submit_load_task(keys, dst_objs)
        wrapper: reserve_write(tmp_keys, byte layout)          # temp buffer
        wrapper: inner.submit_load_task(keys, tmp_objs)
        -> [inner load_event_fd]
        wrapper: serde.submit_deserialize(tmp_objs, dst_objs)  # async
        -> [deserialize_event_fd]
        wrapper: finish_write + delete tmp_keys
    caller: query_load_result(id) → per-key bitmap

The ``SerdeProcessor`` owns two event fds (one per direction);
``SerdeL2AdapterWrapper``'s internal thread polls them along with the
inner adapter's store / load fds. The controllers poll only the
wrapper's own store / load fds — they never see the serde fds or the
inner's fds. Temp buffers are always allocated through ``L1Manager``
so they count against L1 memory accounting.

If **any** key's temp allocation fails, the whole submitted task
fails (``pop_completed_store_tasks`` returns ``False`` for the store
path; ``query_load_result`` returns an all-zeros bitmap for the load
path). This all-or-nothing policy keeps the failure semantics
identical to a raw L2 adapter — controllers need no special handling
for "partial serde failure."


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
``SerdeL2AdapterWrapper`` takes care of temp-buffer allocation, lock
transitions, and chaining into the inner adapter — so custom serdes
never need to touch L1Manager, event fds, or the controllers directly.


Implementation notes
--------------------

- **Temp buffer size.** The temp byte buffer is allocated at exactly
  ``serializer.estimate_serialized_size(layout)`` bytes. Your estimate
  must be an upper bound on the actual serialized output — include any
  safety margin directly in the estimate (e.g., the built-in fp8
  serializer returns ``1.5 * num_elements``).
- **Homogeneous batches.** Every ``submit_store_task`` /
  ``submit_load_task`` passed to a serde-wrapped adapter must carry
  MemoryObjs that share a single ``(shape, dtype)`` — the wrapper sizes
  temp buffers from ``objects[0]``. The store controller already
  shape-groups keys per submission; the prefetch controller uses one
  layout per request. The wrapper raises ``ValueError`` if the
  invariant is violated.
- **Failure handling.** If any step fails (temp alloc, serialize,
  inner store / load, deserialize), the whole wrapped task fails —
  partial success within one submit is not reported. This keeps the
  coarse-grained success semantic of ``L2AdapterInterface`` intact
  and avoids surprising partial writes.
- **Thread pool.** ``AsyncSerdeProcessor(max_workers=N)`` controls the
  pool size. Serde transforms that release the GIL (e.g., torch ops)
  benefit from ``N > 1``; pure-Python transforms do not.
- **Per-adapter independence.** A single prefetch / store request can
  mix serde-enabled and serde-disabled adapters — each serde-wrapped
  adapter runs its own internal thread and temp-buffer lifecycle.


Example
-------

An end-to-end script that starts an lmcache server with fp8 on a disk
adapter, runs vLLM, clears L1, and re-runs the same request to trigger
the L2 prefetch + fp8 deserialize path lives at
:file:`examples/serde/fp8/`. A pytest-based filesystem round-trip test
(no vLLM required) is at
:file:`tests/v1/distributed/serde/test_serde_fs_e2e.py`.


Design references
-----------------

- :file:`docs/design/v1/distributed/serde/README.md` — the serde
  package (Serializer / Deserializer / SerdeProcessor / factory).
- :file:`docs/design/v1/distributed/l2_adapters/serde_wrapper.md` — the
  ``SerdeL2AdapterWrapper`` adapter that integrates serde into the L2
  path.
