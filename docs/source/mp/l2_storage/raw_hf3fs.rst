raw_hf3fs (RDMA native connector)
=================================

A faster 3FS L2 backend implemented directly on the 3FS **Storage Client
RDMA interface** (from the 3FS source tree), in contrast to the
usrbio-based :doc:`hf3fs`. Data chunks are read/written straight through
the RDMA storage client, so the FUSE filesystem is completely bypassed and
the metadata path is greatly shortened.

Design overview (single-file + offset layout, whole-chunk allocation,
chain-parallel batching, metadata persistence, multi-instance lease,
physical delete): see ``csrc/storage_backends/raw_hf3fs/README.md`` in the
source tree.

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
     - ``8``
     - C++ worker threads for I/O.
   * - ``chunk_size``
     - ``4194304`` (4 MB)
     - Chunk size of the single-file offset layout; forced to >= 4 MB and a
       multiple of 4 MB. Should be <= the typical key size and divide it
       evenly (a key spanning k chunks gets k-chain write parallelism).
       Stored files embed the value in their name
       (``<base_dir>/data_<chunk_size>_store``), so changing the chunk size
       starts a fresh store instead of misreading an old layout.
   * - ``meta_addr``
     - ``RDMA://192.168.1.100:8000``
     - Meta server address.
   * - ``mgmtd_addr``
     - ``RDMA://192.168.1.100:8000``
     - Mgmtd server address.
    * - ``transport_pool_max_connections``
     - ``8``
     - Max connections per I/O worker.
   * - ``max_batch_bytes``
     - ``33554432`` (32 MB)
     - Max bytes per StorageClient batchRead/batchWrite RPC.
   * - ``max_concurrent_requests``
     - ``256``
     - Max concurrent StorageClient requests.
   * - ``max_concurrent_requests_per_server``
     - ``64``
     - Max concurrent requests per storage target (0 = 3FS default, 8).
       Raising this is the key lever for large-key (32 MB) workloads.
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
     - Directory (relative to the 3FS cluster root) holding the
       ``data_<chunk_size>_store`` file. **Each LMCache instance must use
       its own** ``base_dir`` -- see Multi-instance safety below.
   * - ``force_use_tcp``
     - ``false``
     - Must stay ``false``: this backend runs over RDMA and never falls back
       to TCP.
   * - ``dir_permission`` / ``file_permission``
     - ``0755`` / ``0644``
     - Permissions for the ``base_dir`` and the store file.
   * - ``token``
     - ``""``
     - 3FS user token for server-side authentication (supports
       ``file://<path>`` to load from a token file). Only required when the
       meta server enables ``authenticate``.

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

**Persistence & Multi-Instance Safety**

The key -> offset map is persisted inside chunk 0 of the store file (with
extension chunks beyond ~150K entries at the 4 MB chunk size) and replayed
on the first operation of a restarted instance, so keys stored by a
previous process are found without any special recovery step. Deleted keys
are physically punched (per-chunk removal) in the background and their
chunks enter a coalesced free list for reuse.

Two LMCache instances sharing one store file would silently corrupt each
other (both bump pointers allocate the same chunks; the metadata snapshot
overwrites the other instance's keys). The store therefore carries an
owner lease: a **second instance is refused and fails fast** with a loud
error while the owner's heartbeat is younger than 60 s; a clean close
releases the lease for immediate handover; crash residue expires via the
TTL. For parallel multi-instance deployments, give each instance its own
``base_dir``.

**Per-Op Timing Diagnostics**

When the connector is closed it prints a per-operation timing breakdown
(Store / Load / Lookup) to stderr, showing where each op's latency goes
(e.g. RDMA ``batch_io`` vs lookup vs memcpy). This is useful for tuning
``num_workers``, ``max_batch_bytes`` and ``max_concurrent_requests``.

**L2 Bench**

Delete accumulated store files before benchmarking (the bench harness
itself never cleans up, and stale stores degrade Store throughput):

.. code-block:: bash

    rm -f /3fs/stage/lmcache/data_*_store
    lmcache bench l2 \
      --data-size-kb 4096 --num-keys 320 --in-flight 16 \
      --rounds 2 --warmup-rounds 1 --lookup-max-hit-rate 0.0 \
      --l2-adapter '{"type":"raw_hf3fs","num_workers":4,"meta_addr":"RDMA://192.168.1.100:8000","mgmtd_addr":"RDMA://192.168.1.100:8000"}'

**Unit Tests**

The ``raw_hf3fs`` integration tests are gated on 3FS extension availability
and a reachable RDMA storage cluster:

.. code-block:: bash

    RUN_HF3FS_INTEGRATION=1 pytest -m integration \
      tests/v1/distributed/test_hf3fs_l2_adapter_v2.py -v
