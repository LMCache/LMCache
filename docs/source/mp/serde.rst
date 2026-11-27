KV Cache Compression
====================

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
   * - ``turboquant``
     - Compress KV tensors with TurboQuant presets before L2 store and
       reconstruct them on load.
     - ``preset`` (default ``turboquant_k8v4``), ``head_dim`` (optional,
       default 128), ``block_size`` (default 16), ``max_workers`` (thread
       pool size, default 1)
   * - ``aesgcm``
     - AES-GCM authenticated encryption of KV bytes at rest in L2, keyed
       per ``cache_salt``.
     - ``key_provider`` (default ``hkdf``), ``master_key_path`` (required
       for ``hkdf``), ``aes_bits`` (``128`` default, or ``256``),
       ``max_workers`` (thread pool size, default 1)
   * - ``gha``
     - Intel only. 4/8-bit quantization of attention KV with native CPU /
       XPU kernels; MLA and unrecognized groups are stored unchanged.
     - See `GHA serde (Intel)`_.

.. warning::

   **Upgrading fp8 serde with a 1-byte KV cache.** Earlier versions of the
   ``fp8`` serde cast a KV cache that is already 1 byte per element (e.g.
   vLLM's fp8 KV cache, held as ``uint8`` codes) numerically to fp8, which
   changed its bytes; the current version stores it byte for byte. The
   stored payload has no dtype or version header, so the two encodings
   cannot be told apart: invalidate L2 entries written for such caches by
   the earlier serde before upgrading, and do not let old and new LMCache
   servers read and write the same L2 store. fp16 / bf16 / fp32 KV caches
   are not affected.


TurboQuant serde
----------------

TurboQuant serde can be enabled by setting ``"type": "turboquant"`` in the
adapter serde config. If ``preset`` is omitted, TurboQuant serde defaults to
``turboquant_k8v4``.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {
                "type": "turboquant",
                "preset": "turboquant_k8v4",
                "block_size": 16
            }
        }'

Supported presets:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Preset
     - Key path
     - Value path
   * - ``turboquant_k8v4``
     - FP8 key
     - 4-bit value quantization
   * - ``turboquant_4bit_nc``
     - 4-bit MSE key with norm correction
     - 4-bit value quantization
   * - ``turboquant_k3v4_nc``
     - 3-bit MSE key with norm correction
     - 4-bit value quantization
   * - ``turboquant_3bit_nc``
     - 3-bit MSE key with norm correction
     - 3-bit value quantization


GHA serde (Intel)
---------------------

GHA quantizes attention K/V to 4 or 8 bits on the way to L2 and restores
it on load, with native kernels for Intel CPUs and Intel XPUs. It is
**Intel-only** and not built by default: build LMCache with the SYCL (Intel
XPU) backend and ``BUILD_GHA=1``:

.. code-block:: bash

    # oneAPI's icpx must be on PATH (SYCL build profile)
    BUILD_GHA=1 pip install -e . --no-build-isolation

This builds the host-CPU kernels twice, ``lmcache.gha_quant`` (x86-64, AVX2
required) and ``lmcache.gha_quant_avx512`` (AVX-512 F/BW/DQ/VL), plus
``lmcache.gha_quant_xpu``. At import time the serde loads the AVX-512 build
when ``/proc/cpuinfo`` reports all four AVX-512 subsets, and the AVX2 build
otherwise. Set ``LMCACHE_GHA_CPU_ISA=avx2`` to force the AVX2 build;
``avx512`` is ignored (with a warning) on a CPU without AVX-512. On any other
platform the modules are not built and configuring ``"type": "gha"`` raises an
error.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {"type": "gha", "qbit": 4, "device": "xpu"}
        }'

No model geometry (head count / head size) is needed. The scaling depends on
how the engine lays out K/V:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``per_tensor``
     - Split K/V chunks ``[2, L, T, H]``
     - Fused K/V chunks ``[L, T, H]``
   * - ``false`` (default)
     - one scale per token
     - one scale per hidden channel
   * - ``true``
     - one scale per plane
     - one scale per plane

.. list-table:: GHA config fields
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Default
     - Meaning
   * - ``qbit``
     - ``4``
     - ``4`` or ``8`` bits per value
   * - ``per_tensor``
     - ``false``
     - one scale per plane for every chunk (smaller, less accurate)
   * - ``device``
     - ``cpu``
     - where the kernels run: ``cpu`` or ``xpu``
   * - ``rh``, ``asym``, ``precond``
     - ``true``
     - randomized Hadamard transform, asymmetric range, preconditioner
   * - ``num_threads``
     - ``4``
     - OpenMP threads of the CPU kernels
   * - ``kv_fp8_dtype``
     - none
     - ``float8_e4m3fn`` / ``float8_e5m2`` when the engine KV cache is fp8
       (fused layout, ``per_tensor=false`` only)
   * - ``groups``, ``default_group``
     - layout read from the chunk shape
     - per ``object_group_id`` layout: ``cache_category`` and
       ``attention_plane_layout`` (``auto``, ``split_kv``, ``fused_kv`` or
       ``mla``)
   * - ``max_workers``
     - ``1``
     - serde thread-pool size

Any other key is rejected. Store and load must use the same config,
including ``device``: the CPU and XPU kernels pack 4-bit fused payloads
differently, so data written with ``"device": "cpu"`` cannot be read with
``"device": "xpu"`` and vice versa.

By default every group is read from its chunk shape: ``[2, L, T, H]`` is split
K/V and ``[L, T, H]`` is fused K/V. An MLA latent or an indexer cache has the
same ``[L, T, H]`` shape as fused K/V, so **declare such groups explicitly**
or they are quantized as fused K/V:

.. code-block:: json

    "groups": {"1": [{"cache_category": "attention",
                      "attention_plane_layout": "mla"}]}

A chunk that cannot be quantized (MLA, unknown layout, or not smaller once
quantized) is stored unchanged and logged once at ``INFO``.


Encryption serde
----------------

The ``aesgcm`` serde encrypts KV bytes with AES-GCM before they land in L2
and decrypts them on load, so a party who can read the remote storage
(bucket / disk / RESP) cannot recover cache contents. Each tenant's data is
encrypted under a distinct key derived from its ``cache_salt``.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "s3",
            "s3_endpoint": "s3://my-kv-cache.s3.us-west-2.amazonaws.com",
            "s3_region": "us-west-2",
            "serde": {
                "type": "aesgcm",
                "key_provider": "hkdf",
                "master_key_path": "/etc/lmcache/keys/master"
            }
        }'

Use your S3 bucket's virtual-hosted endpoint and its AWS region.
Provide the master key as a file (e.g. a mounted Kubernetes ``Secret``).
The ``hkdf`` provider reads it once at startup and derives a per-``cache_salt``
key via HKDF-SHA256; the master key is never written to L2. ``aes_bits``
defaults to ``128`` (already unbreakable and slightly faster); set ``256`` if a
compliance mandate requires it.

Each encrypted object is bound to the key it was stored under: an object that
was modified, or that is found under another key, fails to load and is
recomputed instead of being served. Objects encrypted by earlier LMCache
versions use an older format, so after an upgrade they fail to load once and
are recomputed.

.. warning::

   The ``hkdf`` provider derives every tenant's key from one shared master
   key, so any server holding the master can decrypt any tenant's data
   ("fleet vs. outside" trust). It is not per-tenant access isolation.



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
        def serialize(self, src, dst, key) -> int:
            # Write serialized bytes into dst; return bytes written.
            # ``key`` is the object's ObjectKey; ignore it unless your
            # transform is keyed per object (e.g. encryption keyed on
            # cache_salt).
            ...

        def estimate_serialized_size(self, layout_desc) -> int:
            # Upper bound on serialized byte size for this layout.
            # Override estimate_serialized_size_for_key(layout_desc, key)
            # too if the size depends on the object's group.
            ...

    class MyDeserializer(Deserializer):
        def deserialize(self, src, dst, key) -> None:
            # Read serialized bytes from src, write into dst (KV-shaped).
            # ``key`` mirrors serialize; ignore it unless keyed per object.
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
- **Raw-byte output.** If your serde writes bytes directly (rather than
  through ``MemoryObj.tensor`` like the quantizers), reach the buffer via
  ``MemoryObj.byte_array`` and **cast it to the native format first**:
  ``memoryview(dst.byte_array).cast("B")``. ``byte_array`` is a
  ctypes-backed view with format ``"<B"``, and CPython does not support
  slice assignment into a non-native format (``dst[i:j] = ...`` raises
  ``NotImplementedError: memoryview: unsupported format <B``). The built-in
  ``aesgcm`` serde is an example.
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

Compression methods
-------------------

Higher-level KV cache compression methods layered on top of the per-adapter
serde mechanism:

.. toctree::
   :maxdepth: 1

   cachegen
