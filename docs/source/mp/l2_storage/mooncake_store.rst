Mooncake Store
==============

An L2 adapter backed by the native C++ Mooncake Store connector.  Uses
`Mooncake <https://github.com/kvcache-ai/Mooncake>`_ for high-performance
distributed KV cache storage with RDMA support.

When Mooncake is configured with ``"protocol": "rdma"``, LMCache must also
have a valid contiguous L1 memory region available.  The distributed storage
manager passes this L1 memory descriptor to the adapter factory automatically
in MP mode.  If the descriptor is missing or invalid, adapter creation fails
with ``ValueError`` instead of silently falling back to a non-RDMA path.

**Prerequisites -- Building with Mooncake support:**

The Mooncake extension is **not** built by default.  You must explicitly
enable it:

.. code-block:: bash

    BUILD_MOONCAKE=1 pip install -e . --verbose

The ``BUILD_MOONCAKE`` environment variable controls compilation:

- ``BUILD_MOONCAKE=1``: Enable the Mooncake C++ extension.
- ``BUILD_MOONCAKE=0``: Force disable (highest priority), even if
  ``MOONCAKE_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``MOONCAKE_INCLUDE_DIR`` for
  backward compatibility.  If ``MOONCAKE_INCLUDE_DIR`` is also unset,
  the extension is skipped.

If the Mooncake headers are not installed in the system include path
(e.g., ``/usr/local/include``), you must point to them explicitly:

.. code-block:: bash

    BUILD_MOONCAKE=1 \
    MOONCAKE_INCLUDE_DIR=/path/to/mooncake/include \
    MOONCAKE_LIB_DIR=/path/to/mooncake/lib \
    pip install -e . --verbose

**LMCache-specific fields:**

- ``num_workers``: Number of C++ worker threads for the shared pool
  (default ``4``, must be > 0).

- ``per_op_workers`` (``dict[str, int]``, optional): A dict mapping lane keys
  to dedicated worker thread counts.  Supported keys:

  - ``"lookup"`` — threads for ``EXISTS`` operations.
  - ``"retrieve"`` — threads for ``GET`` / load operations.
  - ``"store"`` — threads for ``SET`` / put operations.
  - ``"delete"`` — threads for ``DELETE`` operations.

  Operations whose lane key is **not** present in the dict use the
  shared ``num_workers`` pool.  There is no requirement to set all
  keys — you can configure only the lanes that need dedicated pools.

**Mooncake fields:**

All other keys in the JSON config (except ``type``, ``num_workers``,
``per_op_workers``, ``process``, ``serde``, and ``eviction``) are forwarded **as-is** to Mooncake's
``setup_internal(ConfigDict)``.  Refer to the
`Mooncake documentation <https://github.com/kvcache-ai/Mooncake>`_
for available setup keys (e.g., ``local_hostname``,
``metadata_server``, ``master_server_addr``, ``protocol``,
``rdma_devices``, ``global_segment_size``).

**Configuration example:**

.. code-block:: bash

    # Shared pool (default)
    --l2-adapter '{
      "type": "mooncake_store",
      "num_workers": 4,
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp",
      "local_buffer_size": "3221225472",
      "global_segment_size": "3221225472"
    }'

    # Per-operation pools (GET-heavy workload)
    --l2-adapter '{
      "type": "mooncake_store",
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      },
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp"
    }'

For full Mooncake setup instructions (master service, metadata server,
etc.), see `Mooncake <https://github.com/kvcache-ai/Mooncake>`_ .

**RDMA notes:**

- ``protocol: "rdma"`` requires a valid LMCache L1 memory descriptor.
- When using ``protocol: "rdma"``, it is recommended to disable lazy L1
  allocation with ``--no-l1-use-lazy`` so the L1 buffer is fully allocated
  before Mooncake registers it.
- ``protocol: "tcp"`` does not require L1 preregistration.
- If Mooncake RDMA initialization fails at adapter creation time, verify that
  LMCache L1 memory is enabled and that the descriptor has a non-zero pointer
  and size.

Process isolation (opt-in)
--------------------------

Add ``"process": {}`` to the adapter JSON to enable the isolated native worker
and admission guard. Omit the field to retain the in-process implementation.
This mode requires Linux, ``--no-l1-use-lazy``, one named SHM arena, and the
prefetch/store transfer-pin lifecycle. Generate a fresh ``--shm-name`` for every
service start. Serde and anonymous or multiple L1 arenas are rejected.

Process fields are ``operation_timeout`` (8 seconds), ``startup_timeout`` (15),
``kill_timeout`` (10), ``restart_interval`` (2), and ``max_pending`` (16).
The nested ``guard`` object accepts the admission/probe policy, for example:

.. code-block:: json

    {
      "process": {
        "operation_timeout": 8.0,
        "guard": {
          "lookup_timeout": 2.0,
          "cooldown": 15.0,
          "probe_timeout": 2.0,
          "recovery_successes": 2
        }
      }
    }

When remote admission is open, misses and bypassed stores preserve L1 serving.
The first affected request can still wait for a deadline and recomputation.
Recovery requires verified data probes, native readiness and drained old I/O.
Inspect ``resilience.remote_available`` separately from bridge ``is_healthy``.
Do not force-clear/delete objects or close L1 while operations are active.
If worker termination cannot be confirmed, the parent exits with code 78 rather
than releasing buffers that might still be accessed.

Inference client deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a checkout that includes the process factory and its transfer-pin
dependencies. Build the native extensions against the Mooncake SDK installed in
the same runtime. Copying Python adapter files into an arbitrary image does not
establish native ABI or inference-engine compatibility.

An inference client can use an existing shared memory pool without contributing
another storage segment: set ``global_segment_size`` to ``"0"``. The separate
Mooncake storage service owns and sizes that pool. The client's ``--l1-size-gb``
still reserves local LMCache memory.

The following example uses documentation-only IP addresses. Replace the client
address, Master, metadata endpoint and RDMA device. Set ``CHUNK_SIZE`` to a size
validated for the model's KV layout; hybrid models can require specific values.

.. code-block:: bash

    : "${CHUNK_SIZE:?Set the model-validated chunk size}"
    SHM_NAME=$(uv run --no-project --python .venv/bin/python python -c \
      'import uuid; print("lmcache_" + uuid.uuid4().hex)')
    ADAPTER='{
      "type": "mooncake_store",
      "local_hostname": "192.0.2.10",
      "master_server_addr": "192.0.2.20:50051",
      "metadata_server": "http://192.0.2.20:18080/metadata",
      "protocol": "rdma",
      "rdma_devices": "mlx5_0",
      "global_segment_size": "0",
      "local_buffer_size": "1073741824",
      "tenant_id": "model-revision-kv-layout-v1",
      "process": {
        "operation_timeout": 8.0,
        "guard": {"lookup_timeout": 2.0, "cooldown": 15.0}
      }
    }'
    uv run --no-project --python .venv/bin/python lmcache server \
      --host 127.0.0.1 --port 15670 \
      --http-host 127.0.0.1 --http-port 18081 \
      --l1-size-gb 128 --shm-name "$SHM_NAME" --no-l1-use-lazy \
      --chunk-size "$CHUNK_SIZE" --separate-object-groups \
      --l2-store-policy default --l2-prefetch-policy retain \
      --l2-prefetch-inflight-watermark 0.5 --l2-adapter "$ADAPTER"

After the local cache server is ready, start vLLM with its model-specific options
and this connector configuration. Keep the model and cache processes on a host
with sufficient RAM, SHM capacity, GPU resources and RDMA access.

.. code-block:: bash

    : "${MODEL_PATH:?Set the model path}"
    uv run --no-project --python .venv/bin/python vllm serve "$MODEL_PATH" \
      --host 127.0.0.1 --port 8000 \
      --kv-transfer-config '{
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
          "lmcache.mp.host": "tcp://127.0.0.1",
          "lmcache.mp.port": 15670,
          "lmcache.mp.mq_timeout": 300.0
        }
      }'

For shutdown or replacement, stop model admission and drain/stop vLLM before
closing LMCache. Its GPU transfers must finish while cache resources still
exist. Let LMCache drain controllers and reap the native worker before replacing
its arena. A supervisor should preserve this order and retain diagnostics when
shutdown cannot be confirmed; it must not automatically remove an old SHM arena
or suppress an exit-78 failure to force a restart.

Use the management ``/status`` endpoint to distinguish local service health from
``storage_manager.l2_adapters[].resilience.remote_available``. An open remote
circuit can coexist with healthy local inference. Restarting the entire service
on every L2 outage would discard that fallback. Query deadlines are not an
end-to-end latency guarantee: an affected request may need recomputation, and
admitted data operations retain their pins until native access has stopped.

Before rollout, verify a normal request, a local cache hit, an L2 hit with an
empty local cache, and recovery after a fault in an isolated test worker or
test backend. Record the source revision, native SDK, framework/CUDA versions,
image digest and profile used. Do not pause a shared production Master or clear
production caches to run these tests.
