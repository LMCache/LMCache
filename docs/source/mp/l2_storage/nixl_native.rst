Native NIXL
===========

``nixl_native`` is LMCache's optional C++ NIXL L2 adapter. It registers the
complete L1 DRAM arena once per worker and submits each LMCache tile as one
NIXL descriptor-list transfer.

The backend is the single storage selector:

.. code-block:: text

   nixl_native
     -> backend selects a NIXL plugin
     -> the plugin's FILE_SEG or OBJ_SEG capability selects storage semantics
     -> backend_params are passed to that plugin

``POSIX`` advertising ``FILE_SEG`` and ``OBJ`` advertising ``OBJ_SEG`` are the
reference configurations below. Another plugin can be substituted only when it
supports ``DRAM_SEG`` and exactly one of those storage segments. The connector
checks those capabilities when it starts and rejects ambiguous plugins.

Build and runtime requirements
------------------------------

The minimum supported NIXL version is 1.3.0. A source installation must install
the public C++ headers as well as ``libnixl``. Build LMCache with:

.. code-block:: bash

   export BUILD_WITH_NIXL=1
   export NIXL_INCLUDE_DIR=/opt/nixl/include
   export NIXL_LIBRARY_DIR=/opt/nixl/lib
   uv pip install -e . --no-build-isolation

``NIXL_PREFIX=/opt/nixl`` can replace the two directory variables when the
installation uses conventional ``include`` and ``lib`` directories. The build
produces ``lmcache.lmcache_nixl`` as an isolated C++20 extension. A normal build
without NIXL headers and libraries does not build or import this extension.

At runtime, the dynamic NIXL plugins must be discoverable. NIXL normally looks
in the ``plugins`` directory next to ``libnixl``. Set ``NIXL_PLUGIN_DIR`` when
the plugins are installed elsewhere, and ensure the dynamic linker can find
``libnixl`` (for example with the system linker cache or ``LD_LIBRARY_PATH``).

Configuration reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Field
     - Default
     - Meaning
   * - ``type``
     - required
     - Must be ``"nixl_native"``.
   * - ``backend``
     - required
     - Uppercase NIXL plugin identifier, such as ``POSIX`` or ``OBJ``. Its
       advertised ``FILE_SEG`` or ``OBJ_SEG`` capability selects the strategy.
   * - ``backend_params``
     - ``{}``
     - String-to-string map passed to NIXL. A ``FILE_SEG`` backend also requires
       ``file_path`` and accepts ``use_direct_io`` and ``shard_dirs``.
   * - ``num_workers``
     - ``4``
     - Positive worker count. Each worker owns one NIXL agent, backend handle,
       and registration of the full L1 arena.
   * - ``max_capacity_gb``
     - ``0``
     - Non-negative capacity for LMCache byte accounting. Zero disables the
       aggregate capacity signal.

The adapter rejects missing L1 memory, invalid backend names, backends that
advertise neither or both storage segments, non-string backend parameters, and
non-positive worker counts during startup. An inferred object strategy rejects
eviction configuration because NIXL 1.3 does not expose object deletion.

POSIX FILE example
------------------

Create a writable directory and start the MP server:

.. code-block:: bash

   install -d -m 0750 /data/lmcache/l2

   lmcache server \
       --host 0.0.0.0 --port 5555 \
       --l1-size-gb 32 \
       --l2-adapter '{
         "type": "nixl_native",
         "backend": "POSIX",
         "backend_params": {
           "file_path": "/data/lmcache/l2",
           "use_direct_io": "false"
         },
         "num_workers": 4,
         "max_capacity_gb": 100
       }'

Stores write unique temporary files, complete the NIXL transfer, call
``fsync``, and publish with an atomic no-replace operation in the same
filesystem. The store path does not perform an existence query; the caller is
expected to query first. If another writer wins the publication race, an
existing file of the expected size is accepted, while a different size is an
error. A failed batch removes its unpublished temporary files and rolls back
files published by that batch.

The submission path also performs no duplicate-key scan. Duplicate serialized
keys in one batch violate the caller contract and must be removed upstream.

Files persist when the connector closes. Lookup queries NIXL using the complete
deterministic path. Load opens each hit and requires its size to equal the
destination size, allowing a mixed batch to load valid files while reporting a
truncated file as a miss. Delete removes the deterministic file.

Set ``use_direct_io`` to ``"true"`` only when the filesystem and L1 layout
support direct I/O. Both buffer address and byte length must be aligned to the
larger of the L1 alignment and filesystem block size. Misaligned operations fail
instead of silently using buffered I/O. ``shard_dirs: "true"`` stores files
under two hash-prefix directories; choose that layout before populating a cache.

OBJ OBJECT example
------------------

Inject credentials through the AWS environment or another credential provider
supported by the AWS SDK. Do not put credentials in ``backend_params`` because
configuration can be displayed by operational tooling.

.. code-block:: bash

   export AWS_ACCESS_KEY_ID='<access-key-from-secret-store>'
   export AWS_SECRET_ACCESS_KEY='<secret-key-from-secret-store>'

   lmcache server \
       --host 0.0.0.0 --port 5555 \
       --l1-size-gb 32 \
       --l2-adapter '{
         "type": "nixl_native",
         "backend": "OBJ",
         "backend_params": {
           "bucket": "example-lmcache-bucket",
           "endpoint_override": "s3.example.com:9000",
           "scheme": "https",
           "region": "us-east-1",
           "use_virtual_addressing": "false"
         },
         "num_workers": 4
       }'

The tested NIXL 1.3 OBJ parameters are ``bucket``, ``endpoint_override``,
``scheme``, ``region``, ``use_virtual_addressing``, ``req_checksum``, and
``resp_checksum``. Consult the selected plugin's documentation before adding
other parameters.

OBJECT stores are unconditional whole-object writes. Store completion means
the NIXL request reached its terminal success state; the connector does not
perform an existence preflight. Query uses the plugin's object query API.
NIXL 1.3 returns existence but not object length, so the connector cannot reject
an oversized object before issuing a ranged read. Partial writes are impossible
through this adapter because every descriptor starts at object offset zero and
covers the complete LMCache object.

.. warning::

   NIXL 1.3 has no supported object-deletion API. ``nixl_native`` therefore
   reports ``supports_delete: false`` and rejects eviction configuration for
   ``OBJECT``. Objects remain until external bucket lifecycle or administrative
   tooling removes them.

Persistent identities
---------------------

Names follow ``nixl_store_dynamic``. A model slash becomes ``--`` and all
``ObjectKey`` identity fields are retained:

.. code-block:: text

   <safe-model>_<kv-rank-8hex>_<object-group-hex>_<chunk-hash-hex>[@salt].bin

For example, model ``org/model``, rank 42, object group 7, hash ``00112233``,
and salt ``tenant`` becomes:

.. code-block:: text

   org--model_0000002a_7_00112233@tenant.bin

This format is intentionally incompatible with ``fs_native``'s existing
``.data`` files. With ``shard_dirs: "true"``, the example lives below
``00/11/``. For object storage those separators form an object-key prefix; for
file storage they are directories beneath ``file_path``.

Benchmarking
------------

Use the public L2 benchmark so all adapters receive equivalent arena-backed
buffers. The following command verifies a POSIX round trip:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"nixl_native","backend":"POSIX","backend_params":{"file_path":"/data/lmcache/bench","use_direct_io":"false"},"num_workers":4}' \
       --num-keys 32 --in-flight 4 --data-size-kb 256 \
       --l1-align-bytes 4096 --warmup-rounds 2 --rounds 10 \
       --no-skip-verify

Run the same command with ``fs_native`` on the same filesystem, payload size,
worker count, and cache state. The existing Python ``nixl_store`` and
``nixl_store_dynamic`` adapters should also be compared after confirming that
their prepared-descriptor index contract is compatible with the benchmark's
arena offsets. On the current development tree with NIXL 1.3.1, both Python
adapters reject this harness with ``makeXferReq local index out of range``; do
not report their failed runs as performance results.

Record the NIXL version, filesystem/device, direct-I/O setting, CPU allocation,
and whether the page cache was warm. OBJ results must also record the
object-store implementation, network topology, request size, and non-secret
endpoint settings.

Troubleshooting
---------------

``lmcache_nixl`` cannot be imported
   Rebuild with ``BUILD_WITH_NIXL=1`` and verify that ``NIXL_INCLUDE_DIR``
   contains ``nixl.h`` and ``NIXL_LIBRARY_DIR`` contains ``libnixl.so``.

Plugin discovery or backend creation fails
   Verify NIXL 1.3 or newer, set ``NIXL_PLUGIN_DIR`` to the installed plugin
   directory, and confirm that the requested plugin supports ``DRAM_SEG`` and
   exactly one of ``FILE_SEG`` or ``OBJ_SEG``.

Buffer is outside the registered L1 arena
   The factory must receive the same L1 arena that owns every submitted
   ``MemoryObj``. The connector deliberately does not register arbitrary
   process memory.

Direct-I/O alignment fails
   Use ``use_direct_io: "false"`` first. For direct I/O, align the L1 base,
   object address, and object length to the required filesystem block size.

OBJ authentication or lookup fails
   Check the credential provider, bucket, endpoint, scheme, region, and path
   versus virtual addressing choice. Status output includes backend and storage
   type but never includes ``backend_params`` or credential values.
