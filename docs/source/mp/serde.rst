L2 Serde (Serialization / Deserialization)
==========================================

LMCache supports a **per-adapter serde** that transforms KV cache data on
its way to and from an L2 adapter. Typical uses: quantization (shrink
storage footprint), compression, encryption.

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
if serde did not exist (no extra allocations, no extra threads).


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


Notes
-----

- **Buffer size.** ``estimate_serialized_size(layout)`` must return an
  upper bound on the actual serialized output — include any safety
  margin directly in the estimate (e.g., the built-in fp8 serializer
  returns ``1.5 * num_elements``).
- **Failure handling.** If any step fails (serialize, store, load, or
  deserialize), the whole submitted batch is reported as failed —
  partial success within one batch is not surfaced. Failed keys are
  cleaned up automatically.
- **Thread pool.** ``AsyncSerdeProcessor(max_workers=N)`` controls the
  pool size. Transforms that release the GIL (e.g., torch ops)
  benefit from ``N > 1``; pure-Python transforms do not.


Example
-------

An end-to-end script that starts an lmcache server with fp8 on a disk
adapter, runs vLLM, clears L1, and re-runs the same request to trigger
the L2 prefetch + fp8 deserialize path lives at
:file:`examples/serde/fp8/`. A pytest-based filesystem round-trip test
(no vLLM required) is at
:file:`tests/v1/distributed/serde/test_serde_fs_e2e.py`.
