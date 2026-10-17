Hf3fs
======================

An L2 adapter backed by the native C++ 3FS connector. Uses the 3FS Usrbio
API for high-performance distributed KV cache storage.

**Prerequisites**

- **3FS Usrbio**
  3FS Usrbio header (``hf3fs_usrbio.h``) and lib (``libhf3fs_api_shared.so``)
  are required when compiling the LMCache hf3fs extension.  We recommend
  installing them from source:

  .. code-block:: bash

      git clone https://github.com/deepseek-ai/3fs
      cd 3fs
      git submodule update --init --recursive
      ./patches/apply.sh
      Install dependencies
      cmake --build build -j 32
      cmake --install build

  See `3FS Build <https://github.com/deepseek-ai/3FS/blob/main/README.md#build-3fs>`_
  for the full build instructions.

  If 3FS Usrbio header or lib are not installed in the system path (e.g.,
  ``/usr/local/include``, ``/usr/local/lib``), you must set
  ``HF3FS_INCLUDE_DIR`` and ``HF3FS_LIB_DIR`` to point to them explicitly:

  .. code-block:: bash

      HF3FS_INCLUDE_DIR=/path/to/usrbio/include \
      HF3FS_LIB_DIR=/path/to/usrbio/lib \

- **libabsl-dev**

  The connector uses Abseil containers (``sharded_flat_hash_map``,
  ``flat_hash_map``) from the system library.  On Debian/Ubuntu:

  .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y libabsl-dev

**Build**

The LMCache 3FS extension is **not** built by default.  You must explicitly enable it:

.. code-block:: bash

    BUILD_HF3FS=1 \
    pip install -e . --verbose

The **BUILD_HF3FS** environment variable controls compilation:

- ``BUILD_HF3FS=1``: Enable build 3FS C++ extension.
- ``BUILD_HF3FS=0``: Force disable (highest priority), even if
  ``HF3FS_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``HF3FS_INCLUDE_DIR``. If
  ``HF3FS_INCLUDE_DIR`` is also unset, the 3FS extension is skipped.

**LMCache-Specific Fields**

- ``num_workers`` (int, default ``4``): C++ worker threads for I/O.
- ``per_op_workers`` (``dict[str, int]``, optional): A dict mapping lane keys
  to dedicated worker thread counts.  Supported keys:

  - ``"lookup"`` — threads for ``EXISTS`` operations.
  - ``"retrieve"`` — threads for ``GET``/load operations.
  - ``"store"`` — threads for ``SET``/put operations.
  - ``"delete"`` — threads for ``DELETE`` operations.

  Operations whose lane key is **not** present in the dict use the
  shared ``num_workers`` pool.  There is no requirement to set all
  keys — you can configure only the lanes that need dedicated pools.

**HF3FS Fields**

- ``mount_point`` (str, required): 3FS mount point directory.
- ``base_paths`` (str, required): Comma-separated subdirectories under
  ``mount_point``.
- ``ior_entries`` (int, default ``256``, range ``[128, 1024]``): Max
  concurrent requests per Ior.
- ``io_depth`` (int, default ``0``, range ``[-128, 128]``): Batch control
  parameter.
- ``numa_id`` (int, default ``-1``): NUMA node ID (``-1`` = current node).
- ``iov_size`` (int, default ``209715200`` (200 MB), range
  ``[104857600, 2147483648]``): Per-thread I/O buffer in bytes.
- ``enable_key_buffer`` (bool, default ``True``): Enable the in-memory
  key buffer for accelerated exists lookups.  When enabled, exist lookups
  respond from a hash set instead of issuing remote 3FS metadata calls.
  This dramatically reduces lookup latency.

**Configuration Example**

.. code-block:: bash

    --l2-adapter '{
      "type": "hf3fs",
      "mount_point": "/3fs/stage",
      "base_paths": "/3fs/stage/path1,/3fs/stage/path2",
      "ior_entries": 256,
      "io_depth": 0,
      "numa_id": -1,
      "iov_size": 209715200,
      "enable_key_buffer": true,
      "num_workers": 4,
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      }
    }'


**In-Memory Key Buffer**

When ``enable_key_buffer`` is ``True`` (the default), the connector maintains an
in-memory cache of all key filenames. Lookup operations respond from the cache instead
of issuing remote 3FS metadata calls. This dramatically reduces lookup latency.

- **Trade-offs**
   - **Enable** for workloads with many concurrent lookups (prefetch,batch exists). 
  The latency savings outweigh the memory cost.
   - **Disable**  when memory is constrained.

To disable the buffer cache:

.. code-block:: bash

    --l2-adapter '{
      "type": "hf3fs",
      "mount_point": "/3fs/stage",
      "base_paths": "/3fs/stage/path1",
      "ior_entries": 256,
      "io_depth": 0,
      "numa_id": -1,
      "iov_size": 209715200,
      "enable_key_buffer": false,
    }'


**Unit Tests**

The hf3fs tests are gated on 3FS extension availability and 3FS storage
cluster accessibility:
- 3FS C++ extension missing → config tests only
- 3FS C++ extension present and 3FS storage cluster available → full test suite

To run the full test suite, set the **HF3FS_MOUNT_POINT** environment
variable to the mount point directory of a real 3FS storage cluster:

.. code-block:: bash

    HF3FS_MOUNT_POINT=/3fs/stage \
    pytest tests/v1/distributed/test_hf3fs_l2_adapter.py -v


raw_hf3fs (RDMA native connector)
=================================

A second, faster 3FS L2 backend implemented directly on the 3FS **Storage
Client RDMA interface** (from the 3FS source tree), in contrast to the
usrbio-based ``hf3fs`` above. Data chunks are read/written straight through
the RDMA storage client, so the FUSE filesystem is completely bypassed and
the metadata path is greatly shortened. This matters most for large-batch
random read/write workloads with many small keys, where it delivers
markedly higher throughput.

**Build**

The ``raw_hf3fs`` extension (``lmcache.lmcache_raw3fs``) is **not** built by
default. Enable it by pointing ``HF3FS_3FS_DIR`` at the 3FS source tree and
setting ``BUILD_WITH_RAW_HF3FS=1``:

.. code-block:: bash

    HF3FS_3FS_DIR=/path/to/3FS \
    BUILD_WITH_RAW_HF3FS=1 \
    pip install -e . --no-build-isolation

**Selection**

Use ``"type": "raw_hf3fs"`` in the ``--l2-adapter`` JSON.

**Configuration Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Field
     - Default
     - Description
   * - ``num_workers``
     - ``4``
     - C++ worker threads for I/O.
   * - ``chunk_size``
     - ``33554432`` (32 MB)
     - Chunk / stride size for the single-file offset layout.
   * - ``meta_addr``
     - ``RDMA://192.168.1.100:8000``
     - Meta server address.
   * - ``mgmtd_addr``
     - ``RDMA://192.168.1.100:8000``
     - Mgmtd server address.
   * - ``max_capacity_gb``
     - ``0``
     - Max L2 capacity in GB for usage tracking / eviction (0 = disabled).
   * - ``transport_pool_max_connections``
     - ``4``
     - Max connections per I/O worker.
   * - ``max_batch_bytes``
     - ``33554432`` (32 MB)
     - Max bytes per StorageClient batchRead/batchWrite RPC.
   * - ``max_concurrent_requests``
     - ``32``
     - Max concurrent StorageClient read requests.
   * - ``max_concurrent_requests_per_server``
     - ``0``
     - Max concurrent requests per storage target (0 = 3FS default, 8).
   * - ``per_op_workers``
     - ``{}``
     - Map from op lane (``lookup``/``retrieve``/``store``/``delete``) to a
       dedicated C++ worker count; lanes not listed share ``num_workers``.
       Default: all ops share one pool.
   * - ``config_file``
     - ``$LMCACHE_3FS_CONFIG`` or ``./lmcache_3fs.json``
     - Path to the 3FS cluster config JSON (values act as defaults).
   * - ``cluster_id``
     - ``"stage"``
     - 3FS cluster identity.
   * - ``chain_table_id``
     - ``1``
     - 3FS chain table ID.
   * - ``base_dir``
     - ``"lmcache"``
     - Directory (relative to the 3FS cluster root) holding ``data.store``.
   * - ``force_use_tcp``
     - ``false``
     - Must stay ``false``: this backend runs over RDMA and never falls back
       to TCP.
   * - ``dir_permission`` / ``file_permission``
     - ``0755`` / ``0644``
     - Permissions for the ``base_dir`` and ``data.store`` file.

**Value Precedence** (highest wins)

#. Explicit ``--l2-adapter`` JSON key.
#. Value from ``config_file`` (if present).
#. Hardcoded C++ default.

**Configuration Example**

.. code-block:: bash

    --l2-adapter '{
      "type": "raw_hf3fs",
      "num_workers": 4,
      "meta_addr": "RDMA://192.168.1.100:8000",
      "mgmtd_addr": "RDMA://192.168.1.100:8000",
      "chunk_size": 33554432,
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      }
    }'

**Shared Cluster Config (``lmcache_3fs.json``)**

The ``--l2-adapter`` JSON can also stay minimal by moving the cluster
parameters (``meta_addr``, ``mgmtd_addr``, ``cluster_id``, ``base_dir``, …)
into a shared ``lmcache_3fs.json`` config file. Create it in the working
directory:

.. code-block:: bash

    cat > lmcache_3fs.json <<'EOF'
    {
      "cluster_id": "stage",
      "meta_addr": "RDMA://192.168.1.100:8000",
      "mgmtd_addr": "RDMA://192.168.1.100:8000",
      "chain_table_id": 1,
      "base_dir": "lmcache",
      "force_use_tcp": false,
      "dir_permission": "0755",
      "file_permission": "0644"
    }
    EOF

By default the adapter loads ``./lmcache_3fs.json`` from the current
directory. To point at a different path, either set the ``config_file`` key
in the adapter JSON or the ``LMCACHE_3FS_CONFIG`` environment variable.
When you use the config file, the ``--l2-adapter`` JSON only needs the
backend-specific keys:

.. code-block:: bash

    --l2-adapter '{"type": "raw_hf3fs", "num_workers": 4}'

Value precedence (highest wins): explicit ``--l2-adapter`` JSON key →
value from the config file → hardcoded C++ default.

**Per-Op Timing Diagnostics**

When the connector is closed it prints a per-operation timing breakdown
(Store / Load / Lookup) to stderr, showing where each op's latency goes
(e.g. RDMA ``batch_io`` vs lookup vs memcpy). This is useful for tuning
``num_workers``, ``max_batch_bytes`` and ``max_concurrent_requests``.

**L2 Bench**

.. code-block:: bash

    lmcache bench l2 \
      --data-size-kb 4096 --num-keys 320 --in-flight 16 \
      --rounds 3 --warmup-rounds 1 --lookup-max-hit-rate 0.0 \
      --l2-adapter '{"type":"raw_hf3fs","num_workers":4,"meta_addr":"RDMA://192.168.1.100:8000","mgmtd_addr":"RDMA://192.168.1.100:8000"}'

**Unit Tests**

The ``raw_hf3fs`` integration tests are gated on 3FS extension availability
and a reachable RDMA storage cluster:

.. code-block:: bash

    pytest tests/v1/distributed/test_hf3fs_l2_adapter_v2.py -v -m integration