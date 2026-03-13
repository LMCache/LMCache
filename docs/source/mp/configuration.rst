Configuration Reference
=======================

This page documents every CLI argument accepted by the LMCache multiprocess
server.  Arguments are grouped by the config module that defines them.

.. contents::
   :local:
   :depth: 2

MP Server
---------

Source: ``lmcache/v1/multiprocess/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--host``
     - ``localhost``
     - Host address to bind the ZMQ server.
   * - ``--port``
     - ``5555``
     - Port to bind the ZMQ server.
   * - ``--chunk-size``
     - ``256``
     - Chunk size for KV cache operations (in tokens).
   * - ``--max-workers``
     - ``1``
     - Maximum number of worker threads for handling ZMQ requests.
   * - ``--hash-algorithm``
     - ``blake3``
     - Hash algorithm for token-based operations.
       Choices: ``builtin``, ``sha256_cbor``, ``blake3``.
   * - ``--engine-type``
     - ``default``
     - Cache engine backend type.
       ``default`` uses MPCacheEngine; ``blend`` uses BlendEngineV2
       for cross-request KV reuse.
       Choices: ``default``, ``blend``.

HTTP Frontend
-------------

Source: ``lmcache/v1/multiprocess/config.py``

Only available when running ``http_server.py``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--http-host``
     - ``0.0.0.0``
     - Host to bind the HTTP (FastAPI/uvicorn) server.
   * - ``--http-port``
     - ``8000``
     - Port to bind the HTTP server.

L1 Memory Manager
------------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--l1-size-gb``
     - *required*
     - Size of L1 memory in GB.
   * - ``--l1-use-lazy`` / ``--no-l1-use-lazy``
     - ``True``
     - Enable or disable lazy allocation for L1 memory.
       Pass ``--l1-use-lazy`` to enable (default) or
       ``--no-l1-use-lazy`` to explicitly disable.
   * - ``--l1-init-size-gb``
     - ``20``
     - Initial allocation size (GB) when using lazy allocation.
   * - ``--l1-align-bytes``
     - ``4096``
     - Alignment size in bytes (default 4 KB).

L1 Manager TTLs
----------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--l1-write-ttl-seconds``
     - ``600``
     - Time-to-live for each object's write lock (seconds).
   * - ``--l1-read-ttl-seconds``
     - ``300``
     - Time-to-live for each object's read lock (seconds).

Eviction Policy
---------------

Source: ``lmcache/v1/distributed/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--eviction-policy``
     - *required*
     - Eviction policy.  Currently only ``LRU`` is supported.
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - Memory usage ratio (0.0--1.0) that triggers eviction.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of allocated memory to evict when triggered (0.0--1.0).

L2 Adapters
-----------

Source: ``lmcache/v1/distributed/l2_adapters/config.py``

L2 adapters are configured via repeatable ``--l2-adapter <JSON>`` arguments.
Each JSON object must include a ``"type"`` field that selects the adapter type.
The order of ``--l2-adapter`` arguments determines the adapter order (cascade).

Registered adapter types: ``nixl_store``, ``fs``, ``mock``.

``nixl_store`` -- NIXL-based persistent storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields:

- ``backend`` *(required)*: One of ``POSIX``, ``GDS``, ``GDS_MT``, ``HF3FS``, ``OBJ``.
- ``backend_params`` *(required for file-based backends)*: Dict of string
  key-value pairs.  File-based backends (``GDS``, ``GDS_MT``, ``POSIX``,
  ``HF3FS``) require ``file_path`` and ``use_direct_io``.
- ``pool_size`` *(required)*: Number of storage descriptors to pre-allocate (> 0).

Examples:

.. code-block:: bash

    # POSIX backend (local file system)
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

    # GDS backend (GPU Direct Storage)
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # GDS_MT backend (multi-threaded GDS)
    --l2-adapter '{"type": "nixl_store", "backend": "GDS_MT", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # HF3FS backend (shared file system)
    --l2-adapter '{"type": "nixl_store", "backend": "HF3FS", "backend_params": {"file_path": "/mnt/hf3fs/lmcache", "use_direct_io": "false"}, "pool_size": 64}'

    # OBJ backend (object store -- no file_path needed)
    --l2-adapter '{"type": "nixl_store", "backend": "OBJ", "backend_params": {}, "pool_size": 32}'

``fs`` -- File-system backed storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pure file-system L2 adapter using async I/O.

Fields:

- ``base_path`` *(required)*: Directory for storing KV cache files.
- ``relative_tmp_dir`` *(optional)*: Relative sub-dir for temp files.
- ``read_ahead_size`` *(optional)*: Trigger read-ahead by reading this many bytes first.
- ``use_odirect`` *(optional)*: Bypass page cache via ``O_DIRECT`` (default ``false``).

Examples:

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

``mock`` -- Mock adapter for testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields:

- ``max_size_gb`` *(required)*: Maximum size of the adapter in GB (> 0).
- ``mock_bandwidth_gb`` *(required)*: Simulated bandwidth in GB/sec (> 0).

Example:

.. code-block:: bash

    --l2-adapter '{"type": "mock", "max_size_gb": 256, "mock_bandwidth_gb": 10}'

Multiple adapters (cascade)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``--l2-adapter`` multiple times.  Adapters are used in the order given:

.. code-block:: bash

    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Prometheus Observability
------------------------

Source: ``lmcache/v1/mp_observability/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--disable-prometheus``
     - ``False``
     - Disable Prometheus metrics collection and HTTP server.
   * - ``--prometheus-port``
     - ``9090``
     - Port to expose the Prometheus ``/metrics`` endpoint.
   * - ``--prometheus-log-interval``
     - ``10.0``
     - How often (seconds) to flush accumulated stats to Prometheus.

Telemetry
---------

Source: ``lmcache/v1/mp_observability/telemetry/config.py``

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--enable-telemetry``
     - ``False``
     - Enable the telemetry event system.
   * - ``--telemetry-max-queue-size``
     - ``10000``
     - Maximum events in the telemetry queue before tail-drop.
   * - ``--telemetry-processor``
     - *(none)*
     - Processor spec as JSON (repeatable).  Must include ``"type"`` field.

``logging`` processor
~~~~~~~~~~~~~~~~~~~~~

The built-in processor.  Logs telemetry events via LMCache's logger.

Fields:

- ``log_level``: Log level to use (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
  ``CRITICAL``).  Default is ``DEBUG``.

Examples:

.. code-block:: bash

    --telemetry-processor '{"type": "logging", "log_level": "DEBUG"}'
    --telemetry-processor '{"type": "logging", "log_level": "INFO"}'

vLLM Client Configuration
--------------------------

On the vLLM side, specify the LMCache server host and port via the
``kv_connector_extra_config`` parameter:

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 6000}}'

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``LMCACHE_LOG_LEVEL``
     - Log level for LMCache (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
       Set to ``DEBUG`` to see L2 store activity, prefetch results, etc.
   * - ``PYTHONHASHSEED``
     - Set to a fixed value for reproducible hashing across processes
       (relevant when using ``--hash-algorithm builtin``).

Full Example
------------

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.http_server \
        --host 0.0.0.0 \
        --port 6555 \
        --chunk-size 512 \
        --max-workers 4 \
        --hash-algorithm blake3 \
        --engine-type default \
        --l1-size-gb 100 \
        --l1-use-lazy \
        --l1-init-size-gb 20 \
        --l1-align-bytes 4096 \
        --l1-write-ttl-seconds 600 \
        --l1-read-ttl-seconds 300 \
        --eviction-policy LRU \
        --eviction-trigger-watermark 0.9 \
        --eviction-ratio 0.1 \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}' \
        --prometheus-port 9090 \
        --prometheus-log-interval 10 \
        --enable-telemetry \
        --telemetry-processor '{"type": "logging", "log_level": "DEBUG"}'
